"""
Extract the 16 baseline lexical/structural features from a RAW URL string,
so the trained Iteration-1 model can be used on any new link in real time
(no network call, no WHOIS/DNS lookup -- instant check before a click).
"""
import re
from urllib.parse import urlparse

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "bl.ink", "rb.gy",
}

COMMON_TLDS = [
    ".com", ".net", ".org", ".info", ".biz", ".xyz", ".top", ".club",
    ".online", ".site", ".in", ".co", ".io", ".gov", ".edu", ".me",
]


def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc.split(":")[0]  # strip port if present
    query = parsed.query

    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    qty_params = len([p for p in query.split("&") if p]) if query else 0
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", url))
    is_shortened = domain.lower() in SHORTENERS
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
    import sys, json
    test_url = sys.argv[1] if len(sys.argv) > 1 else "http://bit.ly/free-prize-claim-now"
    print(json.dumps(extract_features(test_url), indent=2))
