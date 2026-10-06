# AI Powered Malicious Link Defuser — Setup & Run

## 1. Install dependencies
```
pip install scikit-learn pandas matplotlib joblib numpy --break-system-packages
```
(drop `--break-system-packages` if you're using a virtualenv/conda env)

## 2. Get the dataset
Download the CSV into the `data/` folder:
```
cd malicious_link_defuser/data
curl -L -o dataset_full.csv https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv
```
(or just paste that URL into your browser and save the file there)

## 3. Train both models
```
cd ../src
python3 train.py
```
This trains Iteration 1 (baseline) and Iteration 2 (refined), saves the models into
`models/`, the evaluation figures into `figures/`, and prints + saves `results.json`.
Takes under a minute for the baseline, a few minutes for the refined model
(300 trees on 111 features) depending on your machine.

## 4. Try the live demo
From the `src/` folder:
```
python3 predict.py "http://some-suspicious-url.com/login?verify=1"
```
Prints a verdict, a risk score, and the top features driving that decision.

## Folder structure expected
```
malicious_link_defuser/
  data/dataset_full.csv      <- you download this (step 2)
  models/                    <- created by train.py
  figures/                   <- created by train.py
  src/train.py
  src/extract_features.py
  src/predict.py
```
