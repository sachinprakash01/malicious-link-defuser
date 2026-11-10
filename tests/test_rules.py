import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from app import (
    app,
    parse_host,
    get_root_domain,
    check_brand_impersonation,
    is_trusted,
    is_private_ip,
    get_prediction_result,
)


def test_youtube_benign_trusted():
    parsed, host = parse_host("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    root = get_root_domain(host)
    is_imp, brand = check_brand_impersonation(host, root)
    assert not is_imp
    assert is_trusted(host, root)


def test_youtube_impersonation_flagged():
    parsed, host = parse_host("http://youtube.com-verify-account.xyz/login.php")
    root = get_root_domain(host)
    is_imp, brand = check_brand_impersonation(host, root)
    assert is_imp
    assert brand == "youtube"


def test_github_io_untrusted_host():
    parsed, host = parse_host("https://evil.github.io/login")
    root = get_root_domain(host)
    assert not is_trusted(host, root)


def test_google_user_content_subdomain_untrusted():
    parsed, host = parse_host("https://docs.google.com/forms/d/abc")
    root = get_root_domain(host)
    assert not is_trusted(host, root)


def test_amazonaws_not_impersonation():
    parsed, host = parse_host("https://aws.amazonaws.com")
    root = get_root_domain(host)
    is_imp, brand = check_brand_impersonation(host, root)
    assert not is_imp


def test_pineapple_not_impersonation():
    parsed, host = parse_host("http://pineapple.com")
    root = get_root_domain(host)
    is_imp, brand = check_brand_impersonation(host, root)
    assert not is_imp


def test_userinfo_hostname_parsing():
    parsed, host = parse_host("http://user:pass@evil.com")
    assert host == "evil.com"


def test_private_ip_branch_not_phishing():
    r = get_prediction_result("http://192.168.1.1/admin")
    assert r["verdict"] != "Malicious / Phishing"
    assert r["verdict"] == "Private / Local Address"


def test_url_length_limit_returns_400():
    client = app.test_client()
    long_url = "http://example.com/" + "a" * 3000
    response = client.post("/check", json={"url": long_url})
    assert response.status_code == 400
