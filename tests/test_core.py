import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
import app as appmod
from extract_features import extract_features


def run(url):
    return appmod.get_prediction_result(url, "baseline")


def test_userinfo_does_not_spoof_host():
    r = run("http://google.com:x@evil.com/login")
    assert r["ioc"]["root_domain"] == "evil.com"
    assert not r["is_authority"]
    assert "safe" not in r["verdict"].lower()
    assert r["level"] != "safe"


def test_domain_length_uses_real_hostname():
    f = extract_features("http://google.com:x@evil.com/login")
    assert f["domain_length"] == len("evil.com")


def test_schemeless_url_with_scheme_in_query():
    r = run("evil.com/r?u=http://x.com")
    assert r["ioc"]["target_host"] == "evil.com"


def test_feature_counts_known_url():
    f = extract_features("http://a.com/b?x=1&y=2")
    assert f["length_url"] == 22
    assert f["qty_dot_url"] == 1
    assert f["qty_slash_url"] == 3
    assert f["qty_equal_url"] == 2
    assert f["qty_and_url"] == 1
    assert f["qty_params"] == 2
    assert f["domain_length"] == 5


@pytest.mark.parametrize("url", [
    "https://s3.amazonaws.com/bucket/file.pdf",
    "https://fonts.googleapis.com/css2?family=Inter",
    "https://purchase.com/",
    "https://rr1.googlevideo.com/videoplayback",
])
def test_brand_false_positives(url):
    r = run(url)
    assert r.get("is_impersonation", False) is False
    assert r["level"] != "malicious"


@pytest.mark.parametrize("url", [
    "http://youtube.com-verify-account.xyz/login.php",
    "http://secure-paypal-account.com/login",
    "http://paypa1.com/signin",
])
def test_brand_impersonation_caught(url):
    assert run(url)["level"] in ("suspicious", "malicious")


@pytest.mark.parametrize("url", [
    "https://evil.github.io/login",
    "https://docs.google.com/forms/d/e/abc123/viewform",
    "https://drive.google.com/file/d/abc123/view",
    "https://something.blogspot.com/p/verify.html",
])
def test_user_content_never_safe(url):
    r = run(url)
    assert not r["is_defused"]
    assert r["level"] != "safe"
    assert "likely safe" not in r["verdict"].lower()


def test_shortener_not_confirmed_phishing():
    r = run("https://bit.ly/3xY8zQp")
    assert r["level"] != "malicious"
    assert r["verdict"] != "Malicious / Phishing"


def test_private_ip_not_malicious():
    r = run("http://192.168.1.1/admin/config.php")
    assert "malicious" not in r["verdict"].lower()
    assert r["level"] == "not_assessed"


def test_score_range_and_fields():
    r = run("https://en.wikipedia.org/wiki/Random_forest")
    assert 0 <= r["risk_score"] <= 100
    assert 0 <= r["raw_ml_score"] <= 100
    assert "malicious" not in r["verdict"].lower()


@pytest.mark.parametrize("url", [
    "https://case.edu/",
    "https://finance.yahoo.com/quote/AAPL",
    "https://www.google.de/",
    "https://www.google.com.au/",
    "https://www.google.com./",
    "https://facebook.github.io/react-native/",
])
def test_legit_not_flagged_malicious(url):
    assert run(url)["verdict"] != "Malicious / Phishing"
    assert run(url)["level"] != "malicious"


def test_open_redirect_on_authority_not_defused():
    r = run("https://www.google.com/url?q=http://evil.com/login")
    assert not r["is_defused"]
    assert r["verdict"] != "Likely Safe"


def test_decimal_ip_detected():
    assert run("http://2130706433/")["anatomy"]["is_ip"] is True


def test_level_field_exists():
    r = run("https://bit.ly/abc")
    assert r["level"] in {"safe", "suspicious", "unverified", "malicious", "not_assessed"}


def test_requirements_complete():
    root = os.path.join(os.path.dirname(__file__), "..")
    txt = open(os.path.join(root, "requirements.txt")).read().lower()
    assert "flask-limiter" in txt
    assert "tldextract" in txt
    assert "scikit-learn==" in txt


def test_bad_input_returns_400():
    c = appmod.app.test_client()
    assert c.post("/check", json={"url": 123}).status_code == 400
    assert c.post("/check", json={"url": ""}).status_code == 400
    assert c.post("/check", json={"url": "http://[::1"}).status_code == 400
