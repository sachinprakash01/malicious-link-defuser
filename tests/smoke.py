import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import app as appmod

CASES = [
    ("https://en.wikipedia.org/wiki/Random_forest", "safe / not malicious"),
    ("https://notebooklm.google.com/notebook/abc123", "not malicious"),
    ("https://case.edu/", "not malicious"),
    ("https://finance.yahoo.com/quote/AAPL", "not malicious"),
    ("https://www.google.de/", "unverified/safe, not malicious"),
    ("https://www.google.com.au/", "not malicious"),
    ("https://www.google.com./", "safe"),
    ("https://facebook.github.io/react-native/", "not malicious"),
    ("https://s3.amazonaws.com/bucket/file.pdf", "not malicious"),
    ("https://purchase.com/", "not malicious"),
    ("http://google.com:x@evil.com/login", "NOT safe, host = evil.com"),
    ("https://www.google.com/url?q=http://evil.com/login", "NOT safe/defused"),
    ("https://evil.github.io/login", "unverified, never safe"),
    ("https://docs.google.com/forms/d/e/abc123/viewform", "unverified, never safe"),
    ("https://bit.ly/3xY8zQp", "suspicious"),
    ("http://2130706433/", "not_assessed (loopback)"),
    ("http://192.168.1.1/admin/config.php", "not_assessed"),
    ("http://8.8.8.8/login.php", "suspicious"),
    ("http://paypa1.com/signin", "malicious or suspicious"),
    ("http://youtube.com-verify-account.xyz/login.php", "malicious"),
    ("http://secure-paypal-account.com/login", "malicious/suspicious"),
    ("evil.com/r?u=http://x.com", "host = evil.com"),
]

for url, expect in CASES:
    try:
        r = appmod.get_prediction_result(url, "baseline")
        print(f"{str(r.get('level','?')):<13} score={str(r['risk_score']):>6} raw={r['raw_ml_score']:>5} "
              f"host={r['ioc']['target_host']:<22} | want: {expect:<32} | {url}")
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e} | {url}")
