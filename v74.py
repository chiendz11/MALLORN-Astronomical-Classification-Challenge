import os
import re
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==================================================================================
class Config:
    PROJECT_NAME = "MALLORN GRAND ENSEMBLE (5 SEEDS)"
    DATA_ROOT = 'data'
    TRAIN_PATH = os.path.join(DATA_ROOT, 'processed_train_v61_reunion.csv')
    TEST_PATH = os.path.join(DATA_ROOT, 'processed_test_v61_reunion.csv')
    
    # DANH SÁCH 5 SEED ĐỂ LOẠI BỎ YẾU TỐ MAY MẮN
    SEEDS = [42, 2024, 777, 101, 88]
    N_FOLDS = 5
    
    IMPORTANT_FEATURES = [
        'g_amplitude', 'r_amplitude', 'i_amplitude', 'z_amplitude',
        'g_rise_time', 'r_rise_time', 'color_g_r_mean', 'color_r_i_mean',
        'g_kurtosis', 'r_kurtosis', 'host_redshift', 'dist_from_center',
        'g_skew', 'r_skew', 'flux_ratio_g_r', 'flux_ratio_r_i',
        'transient_duration', 'max_flux_g', 'max_flux_r',
        'g_decay_time', 'r_decay_time', 'z_decay_time'
    ]

config = Config()

# ==================================================================================
# 2. BEST PARAMETERS (UPDATED V80)
# ==================================================================================
LGB_PARAMS = {
    "learning_rate": 0.007171669428025716,
    "num_leaves": 88,
    "max_depth": 8,
    "min_child_samples": 59,
    "subsample": 0.7907545168704323,
    "colsample_bytree": 0.6199155567075644,
    "reg_alpha": 0.4175264630662017,
    "reg_lambda": 2.7206397245388487,
    "scale_pos_weight": 30.189636663864377,
    "n_estimators": 10000,
    "objective": "binary",
    "boosting_type": "dart",
    "metric": "binary_logloss",
    "verbose": -1
}

XGB_PARAMS = {
    "learning_rate": 0.08254982951861715,
    "max_depth": 6,
    "min_child_weight": 8,
    "gamma": 2.6542978722989075,
    "subsample": 0.7106937946194638,
    "colsample_bytree": 0.6051441346854033,
    "reg_alpha": 0.8494718160773136,
    "reg_lambda": 2.3647565105630317,
    "scale_pos_weight": 23.78764987396849,
    "n_estimators": 10000,
    "objective": "binary:logistic",
    "max_delta_step": 1
}

CAT_PARAMS = {
    "learning_rate": 0.005046791440166934,
    "depth": 9,
    "l2_leaf_reg": 14.017401606643784,
    "bagging_temperature": 0.6426380960735741,
    "random_strength": 3.389881213267569,
    "scale_pos_weight": 11.96217778289131,
    "iterations": 10000,
    "loss_function": "Logloss",
    "verbose": 0,
    "allow_writing_files": False  # JSON là false, Python phải là False
}

# ==================================================================================
# 3. UTILS
# ==================================================================================
def f1_loss_func(weights, preds_list, y_true):
    w = np.array(weights)
    w = w / np.sum(w)
    final_pred = np.zeros_like(preds_list[0])
    for i, pred in enumerate(preds_list): final_pred += w[i] * pred
    prec, rec, _ = precision_recall_curve(y_true, final_pred)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    return -np.max(f1s)

def optimize_weights(preds_list, y_true):
    init_w = [1.0/len(preds_list)] * len(preds_list)
    bounds = [(0, 1)] * len(preds_list)
    result = minimize(f1_loss_func, init_w, args=(preds_list, y_true),
                      method='SLSQP', bounds=bounds,
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    return result.x

def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], precisions[best_idx], recalls[best_idx]

# ==================================================================================
# 4. TRAINING ENGINE (MULTI-SEED)
# ==================================================================================
def run_grand_ensemble():
    start_total = time.time()
    print("\n" + "="*60)
    print(f"🚀 {config.PROJECT_NAME}")
    print("="*60)
    
    # --- LOAD DATA ---
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    cols_drop = ['object_id', 'target', 'SpecType', 'English Translation', 'split', 'EBV']
    features = [c for c in train_df.columns if c not in cols_drop]
    clean_map = {c: regex.sub("_", c) for c in features}
    train_df.rename(columns=clean_map, inplace=True)
    test_df.rename(columns=clean_map, inplace=True)
    final_features = list(clean_map.values())
    
    X = train_df[final_features]
    y = train_df['target'].values
    X_test = test_df[final_features]
    
    print(f"   -> Samples: {len(X)} | Features: {len(final_features)}")
    
    # Global Arrays to store Average Predictions across all seeds
    # Ta sẽ cộng dồn kết quả dự đoán của từng seed vào đây
    oof_grand_lgb = np.zeros(len(X)); test_grand_lgb = np.zeros(len(X_test))
    oof_grand_xgb = np.zeros(len(X)); test_grand_xgb = np.zeros(len(X_test))
    oof_grand_cat = np.zeros(len(X)); test_grand_cat = np.zeros(len(X_test))

    # --- LOOP THROUGH SEEDS ---
    for seed_i, seed in enumerate(config.SEEDS, 1):
        print(f"\n   >> RUNNING SEED {seed_i}/{len(config.SEEDS)} (State: {seed})...")
        
        # Set seed for params
        LGB_PARAMS['random_state'] = seed
        XGB_PARAMS['random_state'] = seed
        CAT_PARAMS['random_seed'] = seed
        
        # Temp arrays for this seed
        oof_lgb = np.zeros(len(X)); test_lgb = np.zeros(len(X_test))
        oof_xgb = np.zeros(len(X)); test_xgb = np.zeros(len(X_test))
        oof_cat = np.zeros(len(X)); test_cat = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            # Train LGB
            m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
            test_lgb += m_lgb.predict_proba(X_test)[:, 1] / config.N_FOLDS

            # Train XGB
            m_xgb = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=100)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
            test_xgb += m_xgb.predict_proba(X_test)[:, 1] / config.N_FOLDS

            # Train CAT
            m_cat = CatBoostClassifier(**CAT_PARAMS)
            m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
            oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
            test_cat += m_cat.predict_proba(X_test)[:, 1] / config.N_FOLDS

        # Cộng dồn vào Grand Arrays
        oof_grand_lgb += oof_lgb / len(config.SEEDS)
        test_grand_lgb += test_lgb / len(config.SEEDS)
        
        oof_grand_xgb += oof_xgb / len(config.SEEDS)
        test_grand_xgb += test_xgb / len(config.SEEDS)
        
        oof_grand_cat += oof_cat / len(config.SEEDS)
        test_grand_cat += test_cat / len(config.SEEDS)
        
        # Check nhanh điểm của Seed hiện tại
        blend = (oof_lgb + oof_xgb + oof_cat)/3
        _, s_f1, _, _ = find_best_threshold(y, blend)
        print(f"      [Seed {seed} Score]: {s_f1:.4f}")

    # --- FINAL OPTIMIZATION ---
    print("\n>>> OPTIMIZING GRAND ENSEMBLE WEIGHTS...")
    preds_list = [oof_grand_lgb, oof_grand_xgb, oof_grand_cat]
    best_weights = optimize_weights(preds_list, y)
    print(f"   -> Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    
    oof_final = (best_weights[0]*oof_grand_lgb + best_weights[1]*oof_grand_xgb + best_weights[2]*oof_grand_cat)
    best_th, best_f1, best_prec, best_rec = find_best_threshold(y, oof_final)
    
    print(f"\n{'='*40}")
    print(f"🏆 GRAND ENSEMBLE CV SCORE (F1): {best_f1:.5f}")
    print(f"THRESHOLD:                      {best_th:.4f}")
    print(f"PRECISION:                      {best_prec:.2%}")
    print(f"RECALL:                         {best_rec:.2%}")
    print(f"{'='*40}")

    # --- SUBMISSION ---
    test_final = (best_weights[0]*test_grand_lgb + best_weights[1]*test_grand_xgb + best_weights[2]*test_grand_cat)
    final_preds = (test_final >= best_th).astype(int)
    print(f"   -> Final Count: {np.sum(final_preds)}")
    
    sub = pd.DataFrame({'object_id': test_df['object_id'], 'prediction': final_preds})
    sample = pd.read_csv(os.path.join(config.DATA_ROOT, 'test_log.csv'))[['object_id']].drop_duplicates()
    sub = sample.merge(sub, on='object_id', how='left').fillna(0)
    sub['prediction'] = sub['prediction'].astype(int)
    
    save_path = os.path.join(config.DATA_ROOT, 'submission_grand_ensemble.csv')
    sub.to_csv(save_path, index=False)
    print(f"✅ Saved to: {save_path}")
    print(f"⏱️ Total Time: {(time.time()-start_total)/60:.1f} min")

if __name__ == "__main__":
    run_grand_ensemble()