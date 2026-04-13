import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────────────
signals = np.load("data/spy_signals.npy").tolist()
signals = [int(s) for s in signals]
print(f"Total signals: {len(signals)}")

# ── 2. Build sliding window features (pure classical) ─────────
# Instead of reservoir states, use raw binary signals directly
WARMUP = 20
WINDOW = 5

X, y = [], []
for i in range(WARMUP, len(signals) - WINDOW):
    X.append(signals[i : i + WINDOW])   # last 5 days of raw signals
    y.append(signals[i + WINDOW])       # next day direction

X = np.array(X)
y = np.array(y)

print(f"Feature matrix shape: {X.shape}")

# ── 3. Train/test split ───────────────────────────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Train SVM ──────────────────────────────────────────────
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
clf.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n=== Classical Baseline (SVM on raw signals) ===")
print(f"Predicting: will SPY go UP or DOWN tomorrow?")
print(f"Test Accuracy:                {acc:.4f} ({acc*100:.1f}%)")
print(f"Baseline (always predict UP): "
      f"{sum(y_test)/len(y_test):.4f} ({sum(y_test)/len(y_test)*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["Down (0)", "Up (1)"],
                             zero_division=0))