import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from pathlib import Path

# ===============================
# LOAD DATA
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]

attack = pd.read_csv(BASE_DIR / "data/processed/ddos_attack_features.csv")
benign = pd.read_csv(BASE_DIR / "data/processed/ddos_benign_features.csv")

attack["Label"] = 1
benign["Label"] = 0

df = pd.concat([attack, benign], ignore_index=True)

X = df.drop(columns=["Label"])
y = df["Label"]

# ===============================
# MODELS (change paths if needed)
# ===============================
MODELS = {
    "Random Forest": BASE_DIR / "ids/model.pkl",
    "LightGBM": BASE_DIR / "ids/lgbm_model.pkl",
    "LGBM + Adv1": BASE_DIR / "ids/lgbm_model_adv.pkl",
    "LGBM + Adv2": BASE_DIR / "ids/lgbm_model_adv2.pkl",
}

# ===============================
# ROC–AUC CURVE
# ===============================
plt.figure(figsize=(8, 6))

for name, path in MODELS.items():
    model = joblib.load(path)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.predict(X)

    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Comparison of IDS Models")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# CONFUSION MATRICES + REPORTS
# ===============================
for name, path in MODELS.items():
    model = joblib.load(path)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"]
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print(f"\n{name} Classification Report:")
    print(classification_report(y, y_pred))
