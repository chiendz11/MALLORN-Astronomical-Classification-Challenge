import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

# =====================
# PATH CONFIG
# =====================
TRAIN_PATH = "processed_train_data.csv"
TEST_PATH = "processed_test_data.csv"
SUBMIT_PATH = "submission.csv"
MODEL_PATH = "models/tde_lgbm_f1_ensemble.pkl"

N_FOLDS = 5
SEEDS = [42, 2024, 777]

os.makedirs("models", exist_ok=True)

# =====================
# LOAD DATA
# =====================
print("\n>>> LOADING DATA")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# =====================
# DROP INVALID / LEAK COLUMNS
# =====================
DROP_COLS = [
    "object_id",
    "target",
    "SpecType",
    "English Translation",
    "split"
]

y = train_df["target"].values

X = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

# =====================
# SANITY CHECK
# =====================
bad_cols = X.select_dtypes(include=["object"]).columns.tolist()
if len(bad_cols) > 0:
    raise ValueError(f"❌ STILL HAVE OBJECT COLUMNS: {bad_cols}")

print(">>> Feature dtype check:")
print(X.dtypes.value_counts())

# =====================
# CLASS IMBALANCE
# =====================
pos = np.sum(y == 1)
neg = np.sum(y == 0)
scale_pos_weight = neg / pos

print(f"\n>>> POS={pos}, NEG={neg}, scale_pos_weight={scale_pos_weight:.2f}")

# =====================
# LIGHTGBM PARAMS (F1-ORIENTED)
# =====================
params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "metric": "binary_logloss",
    "scale_pos_weight": scale_pos_weight,
    "verbosity": -1
}

# =====================
# TRAINING
# =====================
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

models = []

for seed in SEEDS:
    print("\n" + "=" * 30)
    print(f">>> TRAINING SEED {seed}")
    print("=" * 30)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=True)]
        )

        oof_pred[val_idx] += model.predict(X_val) / len(SEEDS)
        test_pred += model.predict(X_test) / (len(SEEDS) * N_FOLDS)

        models.append(model)

# =====================
# THRESHOLD SEARCH (MAX F1)
# =====================
thresholds = np.arange(0.10, 0.51, 0.01)

best_f1 = 0
best_th = 0

for th in thresholds:
    preds = (oof_pred >= th).astype(int)
    f1 = f1_score(y, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"\n>>> BEST OOF F1 = {best_f1:.4f} @ threshold = {best_th:.2f}")

# =====================
# CONFUSION MATRIX
# =====================
final_oof = (oof_pred >= best_th).astype(int)
cm = confusion_matrix(y, final_oof)

print("\n>>> CONFUSION MATRIX (OOF)")
print(cm)

# =====================
# SUBMISSION
# =====================
test_labels = (test_pred >= best_th).astype(int)

submission = pd.DataFrame({
    "object_id": test_df["object_id"],
    "prediction": test_labels
})

submission.to_csv(SUBMIT_PATH, index=False)
print(f"\n>>> Saved {SUBMIT_PATH}")

# =====================
# SAVE MODEL
# =====================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(models, f)

print(f">>> Saved model: {MODEL_PATH}")
