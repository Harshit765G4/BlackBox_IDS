import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
model = joblib.load(BASE_DIR / "ids/rf_model.pkl")

def ids_oracle(features):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    return int(pred)