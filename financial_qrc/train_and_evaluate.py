import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from reservoir import FinancialQRC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────────────
signals = np.load("data/spy_signals.npy").tolist()
signals = [int(s) for s in signals]
print(f"Total signals: {len(signals)}")

# ── 2. Run the quantum reservoir ──────────────────────────────
N_QUBITS = 4
SHOTS    = 5000
WARMUP   = 20    # increased warmup
WINDOW   = 5     # use last 5 reservoir states as features

print(f"Running quantum reservoir (n_qubits={N_QUBITS}, shots={SHOTS})...")
print("This will take a few minutes, please wait...")

res  = FinancialQRC(n_qubits=N_QUBITS)
mean = res.run(signals, shots=SHOTS)

print(f"Reservoir output shape: {mean.shape}")

# ── 3. Build sliding window features ─────────────────────────
# Instead of 1 feature per step, use WINDOW consecutive
# reservoir states as features — gives the SVM more signal
raw = mean.flatten()   # shape: (499,)

X, y = [], []
for i in range(WARMUP, len(raw) - WINDOW):
    X.append(raw[i : i + WINDOW])        # window of reservoir states
    y.append(signals[i + WINDOW])        # next day's direction

X = np.array(X)   # shape: (N, WINDOW)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")
print(f"Label vector shape:   {y.shape}")

# ── 4. Train/test split ───────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 5. Train SVM classifier ───────────────────────────────────
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
clf.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n=== QRC Financial Direction Classifier ===")
print(f"Predicting: will SPY go UP or DOWN tomorrow?")
print(f"Test Accuracy:                    {acc:.4f} ({acc*100:.1f}%)")
print(f"Baseline (always predict UP):     "
      f"{sum(y_test)/len(y_test):.4f} ({sum(y_test)/len(y_test)*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Down (0)", "Up (1)"],
                             zero_division=0))