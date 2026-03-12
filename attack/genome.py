from pathlib import Path
import json

FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Flow IAT Mean",
    "Flow IAT Std",
    "SYN Flag Count",
    "ACK Flag Count"
]

BASE_DIR = Path(__file__).resolve().parents[1]

with open(BASE_DIR / "dataset" / "feature_bounds.json") as f:
    BOUNDS = json.load(f)