# AI Powered Malicious Link Defuser — Setup & Run

## 1. Install dependencies
```
pip install scikit-learn pandas matplotlib joblib numpy --break-system-packages
```
(drop `--break-system-packages` if you're using a virtualenv/conda env)

## 2. Get the dataset
Download the CSV into the `data/` folder:
```
cd data
curl -L -o dataset_full.csv https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv
```
(or just paste that URL into your browser and save the file there)

## 3. Train the model
```
cd src
python train.py
```
This trains the baseline Random Forest model, saves it into `models/`, the evaluation figures into `figures/`, and prints + saves `results.json`.

## 4. Try the CLI demo
From the repository root:
```
python src/predict.py "http://some-suspicious-url.com/login?verify=1"
```
Prints a verdict, a risk score, and the top features driving that decision.

## 5. Run the web app
From the repository root:
```
python src/app.py
```
Opens the SOC Web Console at `http://localhost:5000`.

## 6. Deploy to Render
- **Environment**: Python 3.11+
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn src.app:app`
- **Note**: Ensure `models/rf_baseline.joblib` and `models/baseline_features.joblib` are committed in the repository so the service starts immediately without retraining.

## Folder structure expected
```
data/dataset_full.csv      <- downloaded dataset
models/                    <- trained baseline model & feature list
figures/                   <- evaluation figures
src/app.py                 <- Flask web app & SOC console
src/train.py               <- training script
src/extract_features.py    <- 16-feature lexical extractor
src/predict.py             <- CLI prediction script
src/templates/index.html   <- SOC dashboard UI
tests/test_rules.py        <- test suite
requirements.txt           <- pinned dependencies
```
