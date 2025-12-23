import os
import re
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# 1. CẤU HÌNH
# ==================================================================================
class Config:
    PROJECT_NAME = "MALLORN PRO TUNING V80 (FIXED ESTIMATORS)"
    DATA_ROOT = 'data'
    TRAIN_PATH = os.path.join(DATA_ROOT, 'processed_train_v61_reunion.csv')
    SEED = 42
    N_FOLDS = 5
    N_TRIALS = 50 
    
    # CHIẾN THUẬT: TRẦN CAO + DỪNG SỚM
    FIXED_N_ESTIMATORS = 10000  # Set thật cao
    EARLY_STOPPING_ROUNDS = 100 # Dừng nếu 100 vòng không cải thiện

def load_data():
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {Config.TRAIN_PATH}")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    cols_drop = ['object_id', 'target', 'SpecType', 'English Translation', 'split', 'EBV']
    features = [c for c in train_df.columns if c not in cols_drop]
    clean_map = {c: regex.sub("_", c) for c in features}
    train_df.rename(columns=clean_map, inplace=True)
    return train_df[list(clean_map.values())], train_df['target'].values

def get_best_f1(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    return np.max(f1s)

# ==================================================================================
# 2. OBJECTIVE FUNCTIONS (CÓ EARLY STOPPING)
# ==================================================================================

# --- LIGHTGBM ---
def objective_lgb(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'dart',
        'random_state': Config.SEED,
        'n_estimators': Config.FIXED_N_ESTIMATORS, # CỐ ĐỊNH
        
        # Search Space
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10.0, 40.0),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.SEED)
    f1_scores = []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**param)
        # Early Stopping trong Fit
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(Config.EARLY_STOPPING_ROUNDS, verbose=False)])
        
        # Lưu lại best_iteration để dùng sau này (nếu cần)
        preds = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        f1_scores.append(get_best_f1(y_val, preds))
        
    return np.mean(f1_scores)

# --- XGBOOST ---
def objective_xgb(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': Config.FIXED_N_ESTIMATORS, # CỐ ĐỊNH
        'random_state': Config.SEED,
        'max_delta_step': 1,
        
        # Search Space
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10.0, 40.0),
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.SEED)
    f1_scores = []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**param, early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1] # XGB tự dùng best_iteration
        f1_scores.append(get_best_f1(y_val, preds))
        
    return np.mean(f1_scores)

# --- CATBOOST ---
def objective_cat(trial, X, y):
    param = {
        'loss_function': 'Logloss',
        'iterations': Config.FIXED_N_ESTIMATORS, # CỐ ĐỊNH
        'random_seed': Config.SEED,
        'verbose': 0,
        'allow_writing_files': False,
        
        # Search Space
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 10.0, 40.0),
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.SEED)
    f1_scores = []
    
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = CatBoostClassifier(**param)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS)
        
        preds = model.predict_proba(X_val)[:, 1]
        f1_scores.append(get_best_f1(y_val, preds))
        
    return np.mean(f1_scores)

# ==================================================================================
# 3. MAIN RUN
# ==================================================================================
def run_optimization():
    print("🚀 STARTING OPTIMIZATION WITH EARLY STOPPING...")
    X, y = load_data()
    best_params_collection = {}
    
    # 1. LightGBM
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=Config.SEED))
    study_lgb.optimize(lambda trial: objective_lgb(trial, X, y), n_trials=Config.N_TRIALS)
    # Cập nhật params tĩnh vào kết quả
    lgb_best = study_lgb.best_params.copy()
    lgb_best.update({'n_estimators': Config.FIXED_N_ESTIMATORS, 'objective': 'binary', 'boosting_type': 'dart', 'metric': 'binary_logloss', 'verbose': -1})
    best_params_collection['lgb'] = lgb_best

    # 2. XGBoost
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=Config.SEED))
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=Config.N_TRIALS)
    xgb_best = study_xgb.best_params.copy()
    xgb_best.update({'n_estimators': Config.FIXED_N_ESTIMATORS, 'objective': 'binary:logistic', 'max_delta_step': 1})
    best_params_collection['xgb'] = xgb_best

    # 3. CatBoost
    study_cat = optuna.create_study(direction='maximize', sampler=TPESampler(seed=Config.SEED))
    study_cat.optimize(lambda trial: objective_cat(trial, X, y), n_trials=Config.N_TRIALS)
    cat_best = study_cat.best_params.copy()
    cat_best.update({'iterations': Config.FIXED_N_ESTIMATORS, 'loss_function': 'Logloss', 'verbose': 0, 'allow_writing_files': False})
    best_params_collection['cat'] = cat_best

    # Save
    with open(os.path.join(Config.DATA_ROOT, 'best_params_v80.json'), 'w') as f:
        json.dump(best_params_collection, f, indent=4)
    
    print("\n✅ DONE! Copy JSON output below to your run script:")
    print(json.dumps(best_params_collection, indent=4))

if __name__ == "__main__":
    run_optimization()