import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ids/fnn_model.pth"
SCALER_PATH = BASE_DIR / "ids/fnn_scaler.pkl"

# ===============================
# QUERY BUDGET
# ===============================
MAX_QUERIES = 1000
_query_count = 0

class QueryBudgetExceeded(Exception):
    pass

def reset_budget():
    global _query_count
    _query_count = 0

def get_query_count():
    return _query_count


# ===============================
# MODEL ARCH (must match training)
# ===============================
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


# ===============================
# LOAD MODEL
# ===============================
scaler = joblib.load(SCALER_PATH)

INPUT_DIM = scaler.mean_.shape[0]
model = FNN(INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# ===============================
# ORACLE
# ===============================
def ids_oracle(features):

    global _query_count

    if _query_count >= MAX_QUERIES:
        raise QueryBudgetExceeded("Budget exceeded")

    _query_count += 1

    x = np.array(features).reshape(1, -1)
    x = scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return 1 if prob >= 0.5 else 0