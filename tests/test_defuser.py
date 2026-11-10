import unittest
from urllib.parse import urlparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from extract_features import extract_features, SCHEME_RE
from app import (
    get_root_domain,
    check_brand_impersonation,
    classify_ip_host,
    is_user_content_host,
    get_prediction_result,
    app
)


class TestDefuser(unittest.TestCase):

    def test_url_parsing_credentials_spoof(self):
        """Test: http://google.com:x@evil.com/login must NOT be treated as google.com."""
        test_url = "http://google.com:x@evil.com/login"
        full_url = test_url if SCHEME_RE.match(test_url) else "http://" + test_url
        parsed = urlparse(full_url)
        host = (parsed.hostname or "").lower()
        root_domain = get_root_domain(host)

        self.assertNotEqual(host, "google.com")
        self.assertNotEqual(root_domain, "google.com")
        self.assertEqual(host, "evil.com")
        self.assertEqual(root_domain, "evil.com")

        # Feature extraction check
        feats = extract_features(test_url)
        self.assertEqual(feats["domain_length"], len("evil.com"))

    def test_brand_check_legitimate_domains(self):
        """Test: amazonaws.com, googleapis.com, purchase.com must NOT flag."""
        # amazonaws.com
        flagged, brand = check_brand_impersonation("amazonaws.com", "amazonaws.com")
        self.assertFalse(flagged, f"amazonaws.com was falsely flagged as {brand}")

        # googleapis.com
        flagged, brand = check_brand_impersonation("googleapis.com", "googleapis.com")
        self.assertFalse(flagged, f"googleapis.com was falsely flagged as {brand}")

        # purchase.com
        flagged, brand = check_brand_impersonation("purchase.com", "purchase.com")
        self.assertFalse(flagged, f"purchase.com was falsely flagged as {brand}")

    def test_brand_check_typosquatting(self):
        """Test: paypa1.com MUST flag."""
        flagged, brand = check_brand_impersonation("paypa1.com", "paypa1.com")
        self.assertTrue(flagged)
        self.assertEqual(brand, "paypal")

    def test_brand_check_subdomain_spoof(self):
        """Test: paypal.com-login.xyz MUST flag."""
        flagged, brand = check_brand_impersonation("paypal.com-login.xyz", "com-login.xyz")
        self.assertTrue(flagged)
        self.assertEqual(brand, "paypal")

    def test_user_content_hosts(self):
        """Test: user-content hosts are identified and never return 'Likely Safe'."""
        user_hosts = [
            "docs.google.com",
            "drive.google.com",
            "sites.google.com",
            "forms.google.com",
            "storage.googleapis.com",
            "sample.googleusercontent.com",
            "project.github.io",
            "myblog.blogspot.com",
            "page.notion.site",
            "dropbox.com",
        ]
        for host in user_hosts:
            self.assertTrue(is_user_content_host(host), f"{host} should be a user-content host")

        # Test prediction verdict for user-content URL
        res = get_prediction_result("https://docs.google.com/document/d/12345/edit")
        self.assertNotEqual(res["verdict"], "Likely Safe")
        self.assertIn("user-hosted content", res["verdict"].lower())

    def test_verdict_shortener(self):
        """Test: shortener returns 'Suspicious (hidden destination)', not Malicious."""
        res = get_prediction_result("https://bit.ly/testlink123")
        self.assertEqual(res["verdict"], "Suspicious (hidden destination)")

    def test_verdict_ip_hosts(self):
        """Test: private IP -> 'Not a public link', public IP -> 'Suspicious'."""
        # Private IP
        res_priv = get_prediction_result("http://192.168.1.1/admin")
        self.assertEqual(res_priv["verdict"], "Not a public link")

        res_loop = get_prediction_result("http://127.0.0.1:8080/dashboard")
        self.assertEqual(res_loop["verdict"], "Not a public link")

        # Public IP
        res_pub = get_prediction_result("http://93.184.216.34/login")
        self.assertEqual(res_pub["verdict"], "Suspicious")

    def test_url_length_cap(self):
        """Test: URL over 2048 chars returns 400 error."""
        client = app.test_client()
        long_url = "https://example.com/" + "a" * 2050
        response = client.post("/check", json={"url": long_url})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
