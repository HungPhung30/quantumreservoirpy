import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from reservoir import FinancialQRC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# ── 1. Load data ──────────────────────────────────────────────
signals = np.load("data/spy_signals.npy").tolist()
signals = [int(s) for s in signals]

SHOTS  = 5000
WARMUP = 20
WINDOW = 5
split_ratio = 0.8

results = {}

# ── 2. Sweep over qubit counts ────────────────────────────────
for n_qubits in [2, 3, 4, 5]:
    print(f"\nRunning n_qubits={n_qubits}...")

    res  = FinancialQRC(n_qubits=n_qubits)
    mean = res.run(signals, shots=SHOTS)
    raw  = mean.flatten()

    # Build sliding window features
    X, y = [], []
    for i in range(WARMUP, len(raw) - WINDOW):
        X.append(raw[i : i + WINDOW])
        y.append(signals[i + WINDOW])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = make_pipeline(StandardScaler(),
                        SVC(gamma='auto', kernel='rbf'))
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    naive = sum(y_test) / len(y_test)
    results[n_qubits] = {"qrc": round(acc, 4), "naive": round(naive, 4)}
    print(f"  QRC Accuracy:   {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Naive Baseline: {naive:.4f} ({naive*100:.1f}%)")

# ── 3. Print summary table ────────────────────────────────────
print("\n=== Qubit Sweep Summary ===")
print(f"{'Qubits':<10} {'QRC Acc':<12} {'Naive Baseline':<15} {'Beat Baseline?'}")
print("-" * 50)
for n, v in results.items():
    beat = "✓ YES" if v["qrc"] > v["naive"] else "✗ NO"
    print(f"{n:<10} {v['qrc']:<12} {v['naive']:<15} {beat}")