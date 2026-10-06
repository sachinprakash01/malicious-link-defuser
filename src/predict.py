"""
AI Powered Malicious Link Defuser -- live demo
Usage: python3 predict.py "http://some-url-to-check.com"
"""
import sys
import joblib
import pandas as pd
from extract_features import extract_features

MODEL_PATH = "../models/rf_baseline.joblib"
FEATURES_PATH = "../models/baseline_features.joblib"


def predict(url: str):
    model = joblib.load(MODEL_PATH)
    feature_order = joblib.load(FEATURES_PATH)

    feats = extract_features(url)
    X = pd.DataFrame([[feats[f] for f in feature_order]], columns=feature_order)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probability of being phishing

    importances = model.feature_importances_
    top_idx = importances.argsort()[::-1][:3]
    top_reasons = [feature_order[i] for i in top_idx]

    label = "MALICIOUS / PHISHING" if pred == 1 else "LIKELY SAFE"
    print(f"\nURL: {url}")
    print(f"Verdict: {label}")
    print(f"Risk score: {proba:.1%}")
    print(f"Top signals the model weighs most (globally): {', '.join(top_reasons)}")
    print(f"This URL's values -> " + ", ".join(f"{f}={feats[f]}" for f in top_reasons))
    return label, proba


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <url>")
        sys.exit(1)
    predict(sys.argv[1])
