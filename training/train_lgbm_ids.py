import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
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

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1]),
    "verbose": -1
}

print("[+] Training LightGBM IDS")

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

model = lgb.train(
    params,
    train_data,
    num_boost_round=800,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50)]
)

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(model, BASE_DIR / "ids/lgbm_model.pkl")
print("[✓] LightGBM model saved")