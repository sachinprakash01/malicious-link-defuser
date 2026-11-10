import os
import sys
import re
import socket
import hashlib
import ipaddress
import logging
import concurrent.futures
from urllib.parse import urlparse
import joblib
import pandas as pd
import tldextract
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Add src folder to sys.path if not present
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import extract_features, is_ip_host, parse_host

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "rf_baseline.joblib")
BASELINE_FEATURES_PATH = os.path.join(MODELS_DIR, "baseline_features.joblib")

# Load models and features once at module level
if not os.path.exists(BASELINE_MODEL_PATH) or not os.path.exists(BASELINE_FEATURES_PATH):
    raise FileNotFoundError(f"Model or feature file not found in {MODELS_DIR}.")

MODEL = joblib.load(BASELINE_MODEL_PATH)
FEATURE_ORDER = joblib.load(BASELINE_FEATURES_PATH)

# Module-level ThreadPoolExecutor for DNS resolution
_dns_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# PSL-aware domain extractor (offline bundled snapshot, memory-only cache to avoid permission errors)
_tld_extractor = tldextract.TLDExtract(cache_dir=False, suffix_list_urls=(), include_psl_private_domains=True)

# Verified high-authority platforms (github.io removed)
AUTHORITY_DOMAINS = {
    "youtube.com", "youtu.be", "google.com", "google.co.in", "googlevideo.com",
    "wikipedia.org", "wikimedia.org", "microsoft.com",
    "apple.com", "amazon.com", "amazon.in", "netflix.com", "spotify.com",
    "stackoverflow.com", "nytimes.com", "bbc.com", "cnn.com",
    "cloudflare.com", "openai.com", "khanacademy.org", "coursera.org", "w3schools.com",
    "mozilla.org", "apache.org", "python.org", "nih.gov", "cdc.gov", "who.int"
}

# User-content roots: trust ONLY host == root or "www." + root, never other subdomains
USER_CONTENT_ROOTS = {
    "google.com", "medium.com", "github.com", "microsoft.com", "amazon.com", "wikipedia.org"
}

# Untrusted hosting platforms: never trusted
UNTRUSTED_HOSTS = {
    "github.io", "blogspot.com", "netlify.app", "vercel.app", "web.app",
    "firebaseapp.com", "pages.dev", "herokuapp.com"
}

# Brand domain ownership map
BRAND_ROOTS = {
    "google": {
        "google.com", "google.co.in", "googlevideo.com", "googleusercontent.com",
        "gstatic.com", "googleapis.com", "youtube.com", "youtu.be"
    },
    "youtube": {"youtube.com", "youtu.be", "googlevideo.com"},
    "amazon": {"amazon.com", "amazon.in", "amazonaws.com", "amazon.co.uk"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com"},
    "apple": {"apple.com", "icloud.com"},
    "paypal": {"paypal.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com", "whatsapp.net"},
    "linkedin": {"linkedin.com"},
    "twitter": {"twitter.com", "x.com"},
    "telegram": {"telegram.org", "t.me"},
    "binance": {"binance.com"},
    "coinbase": {"coinbase.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
}

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
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["120 per minute"],
    storage_uri="memory://"
)


def defang_url(url: str) -> str:
    """Standard security analyst IOC defanging to prevent accidental execution."""
    u = url.replace("http://", "hxxp[://]").replace("https://", "hxxps[://]")
    return u.replace(".", "[.]")


def hash_url(url: str) -> str:
    """SHA-256 fingerprint of the URL for SOC threat intelligence cross-referencing."""
    return hashlib.sha256(url.encode()).hexdigest()


def is_private_ip(host: str) -> bool:
    """Checks if host is a private, loopback, or link-local IP address using ipaddress."""
    if not host:
        return False
    try:
        raw = socket.inet_aton(host)
        ip = ipaddress.ip_address(raw)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except (OSError, socket.error):
        pass
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        pass
    return False


def get_root_domain(host: str) -> str:
    """Extracts registered root domain using tldextract with PSL private domains."""
    if not host:
        return ""
    if is_ip_host(host):
        return host.lower()
    ext = _tld_extractor(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower().rstrip(".")
    if ext.suffix:
        return ext.suffix.lower().rstrip(".")
    return host.lower().rstrip(".")


def check_brand_impersonation(host: str, root_domain: str):
    r"""
    Checks if an unverified domain impersonates a protected brand.
    Tokens = set(re.split(r"[.\-_]", host)).
    Flags only if brand is in tokens AND root_domain not in that brand's legit roots.
    """
    if not host:
        return False, None
    tokens = set(re.split(r"[.\-_]", host.lower()))
    for brand, legit_roots in BRAND_ROOTS.items():
        if brand in tokens and root_domain not in legit_roots:
            return True, brand
    return False, None


def is_trusted(host: str, root_domain: str) -> bool:
    """
    Checks if a host/domain should be trusted as a benign authority platform.
    - UNTRUSTED_HOSTS are never trusted.
    - USER_CONTENT_ROOTS only trust host == root or www.+root (never other subdomains).
    - AUTHORITY_DOMAINS are trusted.
    - Restricts .gov / .edu / .gov.in to exact TLD match.
    """
    if not host:
        return False
    host = host.lower().rstrip(".")
    root_domain = root_domain.lower().rstrip(".")

    # Untrusted hosting platforms are never trusted
    for untrusted in UNTRUSTED_HOSTS:
        if host == untrusted or host.endswith("." + untrusted) or root_domain == untrusted:
            return False

    # For user content roots, trust ONLY apex or www subdomain
    if root_domain in USER_CONTENT_ROOTS:
        return host == root_domain or host == f"www.{root_domain}"

    # Check verified authority platforms
    if root_domain in AUTHORITY_DOMAINS or host in AUTHORITY_DOMAINS:
        return True

    # Restrict .gov / .edu / .gov.in to exact TLD match
    if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".gov.in"):
        return True

    return False


def resolve_dns(host: str):
    """Performs rapid non-blocking DNS resolution with 0.8s timeout using ThreadPoolExecutor."""
    try:
        future = _dns_executor.submit(socket.gethostbyname_ex, host)
        ips = future.result(timeout=0.8)[2]
        return True, ips[0] if ips else "Resolved"
    except Exception:
        return False, "Unresolved / Inactive"


def get_prediction_result(url: str, model_choice: str = "baseline"):
    """
    Evaluates URL using the preloaded Random Forest model and hybrid threat intelligence.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")

    if len(url) > 2048:
        raise ValueError("URL length exceeds 2048 characters")

    # 1. Parse URL & Anatomy safely using parse_host
    parsed, host = parse_host(url)
    if not host:
        raise ValueError("URL has an empty or invalid hostname")

    host = host.rstrip(".")
    root_domain = get_root_domain(host)

    # 2. Extract 16 lexical features & run baseline Random Forest
    feats = extract_features(url)
    X = pd.DataFrame([[feats[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

    pred = int(MODEL.predict(X)[0])
    raw_ml_proba = float(MODEL.predict_proba(X)[0][1])
    raw_ml_score = round(raw_ml_proba * 100, 1)

    importances = MODEL.feature_importances_
    top_idx = importances.argsort()[::-1][:3]
    top_reasons = [FEATURE_ORDER[i] for i in top_idx]

    top_features = [
        {
            "name": f,
            "value": feats.get(f, 0),
            "description": FEATURE_DESCRIPTIONS.get(f, ""),
            "importance": round(float(importances[FEATURE_ORDER.index(f)]), 4)
        }
        for f in top_reasons
    ]

    all_features = [
        {
            "name": f,
            "value": feats.get(f, 0),
            "description": FEATURE_DESCRIPTIONS.get(f, ""),
            "importance": round(float(importances[FEATURE_ORDER.index(f)]), 4) if f in FEATURE_ORDER else 0.0
        }
        for f in FEATURE_ORDER
    ]

    # 3. IP and DNS resolution
    is_ip = is_ip_host(host)
    if is_ip:
        resolved, dns_ip = True, host
    else:
        resolved, dns_ip = resolve_dns(host)

    anatomy = {
        "scheme": (parsed.scheme or "http").upper() if parsed else "HTTP",
        "netloc": parsed.netloc if parsed else host,
        "root_domain": root_domain,
        "path": (parsed.path if parsed else "") or "/",
        "query": (parsed.query if parsed else "") or "(none)",
        "is_ip": is_ip,
        "is_shortened": bool(feats.get("url_shortened", 0)),
        "dns_status": f"Active ({dns_ip})" if resolved else "Unresolved / Inactive",
    }

    # 4. Rule evaluation & Defusal Intelligence
    is_impersonation, target_brand = check_brand_impersonation(host, root_domain)
    is_authority = is_trusted(host, root_domain)

    is_defused = False
    defuser_status = "standard"

    # Branch 1: Private IP -> "Private / Local Address", neutral status, not phishing
    if is_private_ip(host):
        verdict = "Private / Local Address"
        risk_score = 0.0
        threat_class = "Private / Local Network Address"
        defuser_status = "safe"
        defuser_action = (
            f"INTERNAL IP DETECTED: Address '{host}' is a non-routable private or local address. "
            "Not phishing."
        )
        mitre = {
            "id": "N/A",
            "name": "N/A",
            "tactic": "N/A"
        }
        triage = {
            "action": "INTERNAL ONLY",
            "badge": "safe",
            "summary": "Private / local network address. Not accessible or routable on public Internet."
        }

    # Branch 2: Brand Impersonation
    elif is_impersonation:
        verdict = "Malicious / Phishing"
        risk_score = 99.5
        threat_class = "Credential Harvesting / Brand Spoofing"
        defuser_action = (
            f"CRITICAL BRAND IMPERSONATION: Link targets protected brand '{target_brand.capitalize()}' "
            f"from unauthorized host '{root_domain}'. Highly malicious phishing lure."
        )
        defuser_status = "danger"
        mitre = {
            "id": "T1656",
            "name": "Impersonation",
            "tactic": "Defense Evasion"
        }
        triage = {
            "action": "BLOCK & SINKHOLE",
            "badge": "danger",
            "summary": "Block at perimeter firewall and secure email gateway immediately. Quarantine incoming sessions."
        }

    # Branch 3: Public IP Host
    elif is_ip:
        verdict = "Suspicious Direct IP Host"
        risk_score = max(raw_ml_score, 65.0)
        threat_class = "Public Direct IP Host"
        defuser_status = "warning"
        defuser_action = (
            "SUSPICIOUS DIRECT IP HOST: Target uses an explicit public IP address rather than a domain name, "
            "a classic evasion technique used in credential harvesters."
        )
        mitre = {
            "id": "T1584.004",
            "name": "Compromise Infrastructure: Server",
            "tactic": "Resource Development"
        }
        triage = {
            "action": "INSPECT & ISOLATE",
            "badge": "warning",
            "summary": "Direct public IP host bypasses standard domain reputation filters. Inspect in sandbox."
        }

    # Branch 4: Trusted Authority Platform
    elif is_authority:
        verdict = "Likely Safe"
        risk_score = min(round(raw_ml_score * 0.025, 1), 2.5) if raw_ml_score >= 50.0 else raw_ml_score
        threat_class = "Verified Benign Platform"
        is_defused = (raw_ml_score >= 50.0)
        defuser_status = "safe"
        defuser_action = (
            f"DEFUSED FALSE POSITIVE: Domain '{root_domain}' is a verified high-trust global authority. "
            f"The raw lexical model score ({raw_ml_score}%) was defused for legitimate application infrastructure."
            if is_defused else
            f"VERIFIED PLATFORM: Domain '{root_domain}' is a verified high-trust platform."
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

    # Branch 5: Shortened URL
    elif feats.get("url_shortened", 0) == 1:
        verdict = "Suspicious (hidden destination)"
        risk_score = max(raw_ml_score, 60.0)
        threat_class = "Obfuscated Redirector"
        defuser_action = (
            "HIGH RISK OBFUSCATION: URL shortening service detected. The true domain, destination path, "
            "and payload are concealed from pre-click inspection."
        )
        defuser_status = "warning"
        mitre = {
            "id": "T1027",
            "name": "Obfuscated Files or Information",
            "tactic": "Defense Evasion"
        }
        triage = {
            "action": "EXPAND & INSPECT",
            "badge": "warning",
            "summary": "Do not deliver to end users without expanding the underlying destination address in sandbox."
        }

    # Branch 6: Default Lexical Random Forest Evaluation
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
            "name": "User Execution: Malicious Link" if pred == 1 else "Clean Unknown Traffic",
            "tactic": "Execution" if pred == 1 else "Permitted Traffic"
        }
        triage = {
            "action": "BROWSER ISOLATION (RBI)" if pred == 1 else "PERMIT WITH MONITORING",
            "badge": "danger" if pred == 1 else "safe",
            "summary": "Render in isolated sandbox before presenting to users." if pred == 1 else "Standard unrated link; permit with normal audit logging."
        }

    ioc = {
        "defanged_url": defang_url(url),
        "sha256": hash_url(url),
        "target_host": host,
        "root_domain": root_domain,
        "dns_resolution": dns_ip if resolved else "Unresolved / Inactive",
    }

    # Level field for diagnostic tests
    level = "not_assessed" if is_private_ip(host) else ("safe" if defuser_status == "safe" else ("malicious" if defuser_status == "danger" else "suspicious"))

    return {
        "url": url,
        "level": level,
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
        "ioc": ioc,
        "top_features": top_features,
        "all_features": all_features,
        "anatomy": anatomy,
        "note": None,
        "model_used": "Random Forest Ensemble (Vrbancic et al. 2020, 88,647 URLs)"
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
@limiter.limit("30 per minute")
def check():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request"}), 400

    url = data.get("url")
    if not isinstance(url, str):
        return jsonify({"error": "Invalid request"}), 400

    url = url.strip()
    if not url:
        return jsonify({"error": "Invalid request"}), 400

    if len(url) > 2048:
        return jsonify({"error": "Invalid request"}), 400

    parsed, host = parse_host(url)
    if not host:
        return jsonify({"error": "Invalid request"}), 400

    try:
        result = get_prediction_result(url)
        return jsonify(result)
    except ValueError:
        return jsonify({"error": "Invalid request"}), 400
    except Exception:
        logging.exception("Error analyzing URL")
        return jsonify({"error": "Analysis failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Malicious Link Defuser Web UI on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
