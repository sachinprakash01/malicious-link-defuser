"""
Extract the 16 baseline lexical/structural features from a RAW URL string,
so the trained Iteration-1 model can be used on any new link in real time
(no network call, no WHOIS/DNS lookup -- instant check before a click).
"""
import re
import socket
from urllib.parse import urlparse

SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "bl.ink", "rb.gy",
}

COMMON_TLDS = [
    ".com", ".net", ".org", ".info", ".biz", ".xyz", ".top", ".club",
    ".online", ".site", ".in", ".co", ".io", ".gov", ".edu", ".me",
]


def parse_host(url: str):
    """
    Parses a URL and extracts its lowercased hostname safely.
    Handles userinfo (e.g. user:pass@evil.com) and catches ValueError.
    Returns (parsed, host).
    """
    if not isinstance(url, str):
        return None, ""
    clean_url = url.strip()
    if not clean_url:
        return None, ""
    full_url = clean_url if SCHEME_RE.match(clean_url) else "http://" + clean_url
    try:
        parsed = urlparse(full_url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None, ""
    return parsed, host


def is_ip_host(host: str) -> bool:
    if not host:
        return False
    try:
        socket.inet_aton(host)
        return True
    except (OSError, socket.error):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except (OSError, socket.error, AttributeError):
        pass
    return False


def extract_features(url: str) -> dict:
    url = url.strip() if isinstance(url, str) else ""
    parsed, domain = parse_host(url)
    domain = domain.rstrip(".")
    query = parsed.query if parsed else ""

    is_ip = is_ip_host(domain)
    qty_params = len([p for p in query.split("&") if p]) if query else 0
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", url))
    is_shortened = domain in SHORTENERS
    qty_tld = sum(url.lower().count(tld) for tld in COMMON_TLDS)

    return {
        "qty_dot_url": url.count("."),
        "qty_hyphen_url": url.count("-"),
        "qty_underline_url": url.count("_"),
        "qty_slash_url": url.count("/"),
        "qty_questionmark_url": url.count("?"),
        "qty_equal_url": url.count("="),
        "qty_at_url": url.count("@"),
        "qty_and_url": url.count("&"),
        "qty_exclamation_url": url.count("!"),
        "length_url": len(url),
        "qty_tld_url": qty_tld,
        "domain_length": len(domain),
        "domain_in_ip": int(is_ip),
        "qty_params": qty_params,
        "email_in_url": int(has_email),
        "url_shortened": int(is_shortened),
    }


if __name__ == "__main__":
    import sys
    import json
    test_url = sys.argv[1] if len(sys.argv) > 1 else "http://bit.ly/free-prize-claim-now"
    print(json.dumps(extract_features(test_url), indent=2))
