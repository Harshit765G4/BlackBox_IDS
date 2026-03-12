import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import joblib
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]

attack = pd.read_csv(BASE_DIR / "dataset/ddos_attack_features.csv")
benign = pd.read_csv(BASE_DIR / "dataset/ddos_benign_features.csv")

attack["Label"] = 1
benign["Label"] = 0

df = pd.concat([attack, benign], ignore_index=True)

X = df.drop(columns=["Label"]).values
y = df["Label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, BASE_DIR / "ids/fnn_scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    ),
    batch_size=256
)

class FNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FNN(X.shape[1]).to(device)

pos_weight = torch.tensor(
    [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("[+] Training FNN IDS")

for epoch in range(40):
    model.train()
    epoch_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs > 0.5).astype(int)

        y_true.extend(yb.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

print("\nClassification Report")
print(classification_report(y_true, y_pred, digits=4))

roc_auc = roc_auc_score(y_true, y_prob)
print("ROC-AUC:", roc_auc)

torch.save(model.state_dict(), BASE_DIR / "ids/fnn_model.pth")
print("[✓] FNN model saved")