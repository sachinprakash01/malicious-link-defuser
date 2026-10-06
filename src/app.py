import os
import sys
import re
import socket
import hashlib
from urllib.parse import urlparse
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src folder to sys.path if not present
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "rf_baseline.joblib")
BASELINE_FEATURES_PATH = os.path.join(MODELS_DIR, "baseline_features.joblib")
REFINED_MODEL_PATH = os.path.join(MODELS_DIR, "rf_refined.joblib")
REFINED_FEATURES_PATH = os.path.join(MODELS_DIR, "refined_features.joblib")

# Multi-part TLDs for correct root domain extraction
MULTI_PART_TLDS = {
    "co.uk", "co.in", "org.uk", "gov.in", "ac.in", "gov.uk", "co.jp", "com.au", "co.nz", "com.br"
}

# Verified high-authority platforms (Tranco/Alexa top global authorities)
AUTHORITY_DOMAINS = {
    "youtube.com", "youtu.be", "google.com", "google.co.in", "googlevideo.com",
    "wikipedia.org", "wikimedia.org", "github.com", "github.io", "microsoft.com",
    "apple.com", "amazon.com", "amazon.in", "linkedin.com", "twitter.com", "x.com",
    "instagram.com", "facebook.com", "reddit.com", "netflix.com", "spotify.com",
    "stackoverflow.com", "medium.com", "nytimes.com", "bbc.com", "cnn.com",
    "cloudflare.com", "openai.com", "khanacademy.org", "coursera.org", "w3schools.com",
    "mozilla.org", "apache.org", "python.org", "nih.gov", "cdc.gov", "who.int"
}

# Protected brand names frequently targeted in phishing impersonation attacks
PROTECTED_BRANDS = [
    "youtube", "google", "paypal", "netflix", "apple", "amazon", "microsoft",
    "facebook", "instagram", "chase", "wellsfargo", "bankofamerica", "binance",
    "coinbase", "whatsapp", "telegram", "twitter", "linkedin"
]

FEATURE_DESCRIPTIONS = {
    "qty_dot_url": "Total number of dots in the URL",
    "qty_hyphen_url": "Total number of hyphens in the URL",
    "qty_underline_url": "Total number of underscores in the URL",
    "qty_slash_url": "Total number of forward slashes in the URL",
    "qty_questionmark_url": "Total number of question marks in query string",
    "qty_equal_url": "Total number of equals signs in query parameters",
    "qty_at_url": "Presence / count of '@' symbols (credential obfuscation)",
    "qty_and_url": "Total number of '&' parameter delimiters",
    "qty_exclamation_url": "Total number of exclamation marks in the URL",
    "length_url": "Total character length of the complete URL",
    "qty_tld_url": "Count of recognized Top-Level Domains (TLD stacking)",
    "domain_length": "Character length of the domain/hostname",
    "domain_in_ip": "Direct IP address used instead of hostname (1=yes, 0=no)",
    "qty_params": "Number of query parameters detected",
    "email_in_url": "Embedded email pattern found in URL (1=yes, 0=no)",
    "url_shortened": "URL shortening service detected (1=yes, 0=no)",
}

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


def defang_url(url: str) -> str:
    """Standard security analyst IOC defanging to prevent accidental execution."""
    u = url.replace("http://", "hxxp[://]").replace("https://", "hxxps[://]")
    return u.replace(".", "[.]")


def hash_url(url: str) -> str:
    """SHA-256 fingerprint of the URL for SOC threat intelligence cross-referencing."""
    return hashlib.sha256(url.encode()).hexdigest()


def get_root_domain(netloc: str) -> str:
    """Extracts the registered root domain (e.g., www.youtube.com -> youtube.com)."""
    parts = netloc.split(":")[0].lower().split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_TLDS:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return netloc.lower()


def resolve_dns(host: str):
    """Performs rapid non-blocking DNS resolution with a 0.8s timeout."""
    try:
        socket.setdefaulttimeout(0.8)
        ips = socket.gethostbyname_ex(host)[2]
        return True, ips[0] if ips else "Resolved"
    except Exception:
        return False, "Unresolved / Inactive"


def check_brand_impersonation(host: str, root_domain: str):
    """
    Checks if an unverified domain is pretending to be a protected brand
    (e.g., youtube.verify-login.xyz or secure-paypal-account.com).
    """
    for brand in PROTECTED_BRANDS:
        if brand in host and not root_domain.startswith(brand + "."):
            return True, brand
    return False, None


def get_prediction_result(url: str, model_choice: str = "baseline"):
    """
    Enterprise Threat Intelligence & Link Defuser Engine:
    Combines the 16 lexical ML Random Forest signals with real-world
    domain reputation, brand impersonation defense, and live DNS resolution.
    """
    model_choice = (model_choice or "baseline").lower()
    note = None

    if "refined" in model_choice:
        note = "Refined model needs WHOIS/DNS lookups not available in this live demo — showing baseline prediction instead."
        model_path = BASELINE_MODEL_PATH
        features_path = BASELINE_FEATURES_PATH
    else:
        model_path = BASELINE_MODEL_PATH
        features_path = BASELINE_FEATURES_PATH

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Model or feature file not found in {MODELS_DIR}.")

    model = joblib.load(model_path)
    feature_order = joblib.load(features_path)

    # 1. Parse URL & Anatomy
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0].lower()
    root_domain = get_root_domain(host)

    # 2. Extract 16 lexical features & run baseline Random Forest
    feats = extract_features(url)
    X = pd.DataFrame([[feats[f] for f in feature_order]], columns=feature_order)

    pred = int(model.predict(X)[0])
    raw_ml_proba = float(model.predict_proba(X)[0][1])
    raw_ml_score = round(raw_ml_proba * 100, 1)

    importances = model.feature_importances_
    top_idx = importances.argsort()[::-1][:3]
    top_reasons = [feature_order[i] for i in top_idx]

    top_features = [
        {
            "name": f,
            "value": feats.get(f, 0),
            "description": FEATURE_DESCRIPTIONS.get(f, ""),
            "importance": round(float(importances[feature_order.index(f)]), 4)
        }
        for f in top_reasons
    ]

    all_features = [
        {
            "name": f,
            "value": feats.get(f, 0),
            "description": FEATURE_DESCRIPTIONS.get(f, ""),
            "importance": round(float(importances[feature_order.index(f)]), 4) if f in feature_order else 0.0
        }
        for f in feature_order
    ]

    # 3. DNS Lookup
    is_ip = bool(feats.get("domain_in_ip", 0))
    resolved, dns_ip = (True, host) if is_ip else resolve_dns(host)

    anatomy = {
        "scheme": (parsed.scheme or "http").upper(),
        "netloc": parsed.netloc,
        "root_domain": root_domain,
        "path": parsed.path or "/",
        "query": parsed.query or "(none)",
        "is_ip": is_ip,
        "is_shortened": bool(feats.get("url_shortened", 0)),
        "dns_status": f"Active ({dns_ip})" if resolved else "Unresolved / Offline"
    }

    # 4. Hybrid Defusal Intelligence Fusion
    is_impersonation, target_brand = check_brand_impersonation(host, root_domain)
    is_authority = (
        root_domain in AUTHORITY_DOMAINS or
        root_domain.endswith(".gov") or
        root_domain.endswith(".edu") or
        root_domain.endswith(".gov.in")
    )

    is_defused = False
    defuser_status = "standard"  # "safe", "danger", "warning", "defused"

    if is_impersonation:
        verdict = "Malicious / Phishing"
        risk_score = 99.5
        threat_class = "Credential Harvesting / Brand Spoofing"
        defuser_action = (
            f"CRITICAL BRAND IMPERSONATION: Link spoofs '{target_brand.capitalize()}' "
            f"from an unauthorized host ('{root_domain}'). Highly malicious phishing lure."
        )
        defuser_status = "danger"
        mitre = {
            "id": "T1656 / T1566.002",
            "name": "Impersonation & Spearphishing Link",
            "tactic": "Initial Access / Defense Evasion"
        }
        triage = {
            "action": "BLOCK & SINKHOLE",
            "badge": "danger",
            "summary": "Block at perimeter firewall and secure email gateway immediately. Quarantine incoming sessions."
        }

    elif is_ip:
        verdict = "Malicious / Phishing"
        risk_score = max(raw_ml_score, 96.0)
        threat_class = "Direct IP Host / Evasion Technique"
        defuser_action = (
            "SUSPICIOUS DIRECT IP HOST: Target uses an explicit IP address rather than a domain name, "
            "a classic evasion technique used in credential harvesters."
        )
        defuser_status = "danger"
        mitre = {
            "id": "T1584.004",
            "name": "Compromise Infrastructure / Direct IP Host",
            "tactic": "Command & Control / Evasion"
        }
        triage = {
            "action": "BLOCK & ISOLATE",
            "badge": "danger",
            "summary": "Direct IP requests bypass standard domain filters. Block outbound connections from enterprise endpoints."
        }

    elif is_authority and resolved:
        verdict = "Likely Safe"
        risk_score = min(round(raw_ml_score * 0.025, 1), 2.5)  # defused to < 2.5%
        threat_class = "Verified Benign Platform"
        is_defused = True
        defuser_status = "safe"
        defuser_action = (
            f"DEFUSED FALSE POSITIVE: Domain '{root_domain}' is a verified high-trust global authority. "
            f"The raw lexical model's high score ({raw_ml_score}%) due to path slashes and query strings "
            f"was defused as standard media/application routing."
        )
        mitre = {
            "id": "BENIGN",
            "name": "Trusted Global Authority Infrastructure",
            "tactic": "Legitimate Traffic"
        }
        triage = {
            "action": "PERMIT / TRUSTED",
            "badge": "safe",
            "summary": "Platform identity validated. Safe for unrestricted endpoint access without quarantine."
        }

    elif feats.get("url_shortened", 0) == 1:
        verdict = "Malicious / Phishing"
        risk_score = max(raw_ml_score, 88.5)
        threat_class = "Obfuscated Redirector"
        defuser_action = (
            "HIGH RISK OBFUSCATION: URL shortening service detected. The true domain, destination path, "
            "and payload are concealed from pre-click inspection."
        )
        defuser_status = "danger"
        mitre = {
            "id": "T1027",
            "name": "Obfuscated / Compressed Information",
            "tactic": "Defense Evasion"
        }
        triage = {
            "action": "EXPAND & INSPECT",
            "badge": "warning",
            "summary": "Do not deliver to end users without expanding the underlying destination address in sandbox."
        }

    elif not resolved:
        verdict = "Suspicious / Dead Domain"
        risk_score = max(raw_ml_score, 82.0)
        threat_class = "Disposable / Inactive Host"
        defuser_action = (
            f"UNRESOLVED HOST: '{host}' could not be resolved via public DNS. "
            "Domain is either dead, ephemeral, or blocked by upstream security resolvers."
        )
        defuser_status = "warning"
        mitre = {
            "id": "T1583.001",
            "name": "Acquire Infrastructure: Domains",
            "tactic": "Resource Development"
        }
        triage = {
            "action": "DENY UNRESOLVED",
            "badge": "warning",
            "summary": "Host has no valid A/AAAA DNS records. Deny connection to prevent sinkhole traps."
        }

    else:
        verdict = "Malicious / Phishing" if pred == 1 else "Likely Safe"
        risk_score = raw_ml_score
        threat_class = "Lexical Anomaly Suspect" if pred == 1 else "Standard Benign Web Link"
        defuser_status = "danger" if pred == 1 else "safe"
        defuser_action = (
            f"LEXICAL ML CLASSIFICATION: Analyzed 16 structural signals across the Random Forest ensemble. "
            f"Calculated threat probability: {raw_ml_score}%."
        )
        mitre = {
            "id": "T1204.001" if pred == 1 else "BENIGN",
            "name": "User Execution: Malicious URL" if pred == 1 else "Clean Unknown Traffic",
            "tactic": "Initial Access" if pred == 1 else "Permitted Traffic"
        }
        triage = {
            "action": "BROWSER ISOLATION (RBI)" if pred == 1 else "PERMIT WITH MONITORING",
            "badge": "danger" if pred == 1 else "safe",
            "summary": "Render in isolated sandbox before presenting to users." if pred == 1 else "Standard unrated link; permit with normal audit logging."
        }

    # Threat vector multi-dimensional breakdown (Expert SOC Radar)
    threat_vectors = {
        "lexical": raw_ml_score,
        "brand": 99.0 if is_impersonation else (0.0 if is_authority else 15.0),
        "infrastructure": 95.0 if is_ip else (85.0 if not resolved else (5.0 if is_authority else 40.0)),
        "trust": 98.0 if is_authority else (5.0 if (is_impersonation or is_ip) else 35.0)
    }

    ioc = {
        "defanged_url": defang_url(url),
        "sha256": hash_url(url),
        "target_host": host,
        "root_domain": root_domain,
        "dns_resolution": dns_ip if resolved else "Failed"
    }

    return {
        "url": url,
        "verdict": verdict,
        "risk_score": risk_score,
        "raw_ml_score": raw_ml_score,
        "threat_class": threat_class,
        "is_defused": is_defused,
        "defuser_status": defuser_status,
        "defuser_action": defuser_action,
        "is_authority": is_authority,
        "is_impersonation": is_impersonation,
        "mitre": mitre,
        "triage": triage,
        "threat_vectors": threat_vectors,
        "ioc": ioc,
        "top_features": top_features,
        "all_features": all_features,
        "anatomy": anatomy,
        "note": note,
        "model_used": "Enterprise Hybrid Link Defuser (Random Forest + Threat Intel Engine)"
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    model_choice = data.get("model", "baseline")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        result = get_prediction_result(url, model_choice)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Malicious Link Defuser Web UI on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
