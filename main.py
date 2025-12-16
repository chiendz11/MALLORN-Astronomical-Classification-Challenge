import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
# Đã sửa lỗi import trong phiên bản trước
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve 

# ==========================================
# 0. CẤU HÌNH & HẰNG SỐ 
# ==========================================
DATA_ROOT = './data/'
TRAIN_LOG_PATH = os.path.join(DATA_ROOT, 'train_log.csv')
TEST_LOG_PATH = os.path.join(DATA_ROOT, 'test_log.csv')
# Đổi tên file submission cuối cùng
SUBMIT_PATH = "submission_lgbm_final_v49.csv" 
MODEL_PATH = "models/tde_lgbm_final_v49.pkl"

R_LAMBDA = {
    'u': 4.239, 'g': 3.303, 'r': 2.285, 'i': 1.698, 'z': 1.263, 'y': 1.061
}
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']
NUM_SPLITS = 20  
N_FOLDS = 5
SEEDS = [42, 2024, 777] 
SEED = 42 
EPS = 1e-6

os.makedirs("models", exist_ok=True)

# ==========================================
# 1. HÀM FEATURE ENGINEERING (GIỮ NGUYÊN)
# ==========================================
# (Các hàm correct_flux, extract_time_features, extract_features, process_dataset giữ nguyên như V4.8)

def correct_flux(df, ebv_map):
    df['EBV'] = df['object_id'].map(ebv_map).fillna(0)
    df['R_val'] = df['Filter'].map(R_LAMBDA)
    correction = np.power(10, 0.4 * df['R_val'] * df['EBV'])
    return df['Flux'] * correction


def extract_time_features(df):
    feats = {}
    time = df['Time (MJD)'].values
    flux = df['Flux_corr'].values

    if len(time) < 3:
        return {}

    peak_idx = np.argmax(flux)
    t_peak = time[peak_idx]

    feats['t_rise'] = t_peak - time.min()
    feats['t_decay'] = time.max() - t_peak

    q25 = np.quantile(time, 0.25)
    q75 = np.quantile(time, 0.75)

    early_flux = flux[time <= q25]
    late_flux = flux[time >= q75]

    feats['early_late_flux_ratio'] = (
        np.mean(early_flux) / (np.mean(late_flux) + EPS)
        if len(early_flux) and len(late_flux) else 0
    )

    feats['n_peak_points'] = np.sum(flux > 0.8 * flux.max())
    return feats


def extract_features(lc_df, log_df):
    features = []

    for oid, df in tqdm(lc_df.groupby('object_id'), desc="Extracting features"):
        obj = {'object_id': oid}

        z = log_df.loc[oid, 'Z'] if oid in log_df.index and not pd.isna(log_df.loc[oid, 'Z']) else 0.0
        
        df = df.copy()
        df['Time_rest'] = df['Time (MJD)'] / (1 + z)

        flux = df['Flux_corr'].values
        ferr = df['Flux_err'].values

        # Global statistics
        obj['flux_mean'] = np.mean(flux)
        obj['flux_std'] = np.std(flux)
        obj['flux_median'] = np.median(flux)
        obj['flux_max'] = np.max(flux)
        obj['flux_min'] = np.min(flux)
        obj['flux_skewness'] = skew(flux)
        obj['flux_amplitude'] = obj['flux_max'] - obj['flux_min']
        obj['burst_ratio'] = obj['flux_max'] / (abs(obj['flux_median']) + EPS)
        obj['flux_mad'] = np.median(np.abs(flux - obj['flux_median']))
        obj['neg_flux_ratio'] = np.mean(flux < 0)

        # SNR features
        snr = flux / (ferr + EPS)
        obj['snr_mean'] = np.mean(snr)
        obj['snr_max'] = np.max(snr)
        weights = 1 / (ferr ** 2 + EPS)
        obj['flux_weighted_mean'] = np.sum(flux * weights) / np.sum(weights)

        # Time-domain features
        time_feats = extract_time_features(df)
        obj.update(time_feats)

        # Rest-frame timescale
        obj['t_rise_rest'] = obj.get('t_rise', 0) / (1 + z)
        obj['t_decay_rest'] = obj.get('t_decay', 0) / (1 + z)

        # Per-filter features
        for f in FILTERS:
            fdf = df[df['Filter'] == f]
            if fdf.empty: continue
            fflux = fdf['Flux_corr'].values
            obj[f'{f}_mean'] = np.mean(fflux)
            obj[f'{f}_max'] = np.max(fflux)
            obj[f'{f}_skewness'] = skew(fflux) if len(fflux) > 2 else 0.0

        # Color features
        pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]
        for f1, f2 in pairs:
            if f'{f1}_mean' in obj and f'{f2}_mean' in obj:
                obj[f'{f1}_minus_{f2}_mean'] = obj[f'{f1}_mean'] - obj[f'{f2}_mean']
            if f'{f1}_max' in obj and f'{f2}_max' in obj:
                obj[f'{f1}_minus_{f2}_max'] = obj[f'{f1}_max'] - obj[f'{f2}_max']

        # Meta
        obj['n_obs'] = len(df)
        obj['n_filters'] = df['Filter'].nunique()

        features.append(obj)
    return pd.DataFrame(features)


def process_dataset(mode='train'):
    cache_file = f"processed_{mode}_data_v3.csv"
    if os.path.exists(cache_file):
        # Nạp cache data đã xử lý
        print(f">>> Loading cached data: {cache_file}")
        return pd.read_csv(cache_file)
    
    # Đoạn này sẽ không chạy trừ khi bạn xóa file cache

# ==========================================
# 2. CUSTOM METRIC (Cho F1/FBeta Score)
# ==========================================

def lgbm_f1_metric(preds, train_data):
    """Custom Metric cho LightGBM để tối ưu F1 Score"""
    labels = train_data.get_label()
    precision, recall, _ = precision_recall_curve(labels, preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + EPS)
    best_f1 = np.max(f1_scores)
    return 'f1_score', best_f1, True

# ==========================================
# 3. CHẠY MODEL LIGHTGBM ENSEMBLE (Tối ưu F1 - V4.9)
# ==========================================

def run_lgbm_optimized():
    # --- Load Data (Dùng cached data) ---
    train_df = pd.read_csv(f"processed_train_data_v3.csv")
    test_df = pd.read_csv(f"processed_test_data_v3.csv")
    
    # =====================
    # PREPARE DATA
    # =====================
    DROP_COLS = [
        "object_id", "target", "SpecType", "English Translation", "split"
    ]
    
    y = train_df["target"].values
    
    X = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

    X = X.fillna(0.0)
    X_test = X_test.fillna(0.0)
    
    print(f"\n>>> Training on {X.shape[1]} features.")

    # =====================
    # CLASS IMBALANCE (Tăng cường Recall)
    # =====================
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    # V4.9: Tăng scale_pos_weight lên 1.1 lần để tăng Recall
    scale_pos_weight = (neg / pos) * 1.1 
    
    print(f"\n>>> POS={pos}, NEG={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # =====================
    # LIGHTGBM PARAMS (Tối ưu hóa Regularization)
    # =====================
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.02, 
        "num_leaves": 40, 
        "max_depth": 7, 
        "min_data_in_leaf": 30, 
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.3, # Tăng L1 (Lasso) một chút
        "lambda_l2": 0.1, # Giữ L2
        "metric": "binary_logloss",
        "scale_pos_weight": scale_pos_weight, # Trọng số mới
        "verbosity": -1,
        "n_jobs": -1,
        "seed": SEED
    }

    # =====================
    # TRAINING & ENSEMBLE
    # =====================
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for seed in SEEDS:
        print("\n" + "=" * 30)
        print(f">>> TRAINING SEED {seed} (V4.9: Higher Recall)")
        print("=" * 30)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Fold {fold}/{N_FOLDS}")

            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            dtrain = lgb.Dataset(X_tr, y_tr)
            dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

            model = lgb.train(
                params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                feval=lgbm_f1_metric,
                callbacks=[lgb.early_stopping(150, verbose=False)]
            )

            oof_pred[val_idx] += model.predict(X_val) / len(SEEDS)
            test_pred += model.predict(X_test) / (len(SEEDS) * N_FOLDS)

    # =====================
    # THRESHOLD SEARCH (MAX F1)
    # =====================
    thresholds = np.arange(0.05, 0.60, 0.01) # Mở rộng phạm vi tìm kiếm

    best_f1 = 0
    best_th = 0

    for th in thresholds:
        preds = (oof_pred >= th).astype(int)
        f1 = f1_score(y, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"\n>>> BEST OOF F1 (V4.9 FINAL) = {best_f1:.4f} @ threshold = {best_th:.2f}")

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


if __name__ == "__main__":
    run_lgbm_optimized()