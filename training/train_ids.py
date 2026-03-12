import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]

attack = pd.read_csv(BASE_DIR / "dataset/ddos_attack_features.csv")
benign = pd.read_csv(BASE_DIR / "dataset/ddos_benign_features.csv")

attack["Label"] = 1
benign["Label"] = 0

df = pd.concat([attack, benign])

X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("[+] Training RF IDS")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, BASE_DIR / "ids/rf_model.pkl")
print("[✓] RF model saved")