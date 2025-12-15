# train.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import config
import matplotlib.pyplot as plt

def train_and_predict():
    # 1. Load dữ liệu đã xử lý (đỡ phải tính lại từ đầu)
    print("Loading processed data...")
    train_df = pd.read_csv("processed_train_data.csv")
    test_df = pd.read_csv("processed_test_data.csv")
    
    # 2. Chuẩn bị Features (X) và Label (y)
    # Loại bỏ các cột không dùng để train
    ignore_cols = ['object_id', 'split', 'target', 'SpecType', 'English Translation']
    features = [c for c in train_df.columns if c not in ignore_cols]
    
    X = train_df[features]
    y = train_df['target']
    
    X_test = test_df[features]
    
    print(f"Training with {len(features)} features on {len(X)} samples.")
    
    # 3. Cấu hình LightGBM
    # is_unbalance=True là chìa khóa để bắt được TDE hiếm
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': -1,
        'is_unbalance': True, 
        'random_state': config.SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 4. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    oof_preds = np.zeros(len(X))          # Lưu kết quả dự đoán trên tập train (để đánh giá)
    test_preds = np.zeros(len(X_test))    # Lưu kết quả dự đoán trên tập test (để nộp bài)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        
        # Train
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
        )
        
        # Predict Validation
        val_prob = clf.predict_proba(X_val)[:, 1]
        
        # Tối ưu ngưỡng (Threshold Optimization)
        # Vì dữ liệu mất cân bằng, ngưỡng 0.5 chưa chắc tốt nhất.
        # Ta thử tìm ngưỡng tốt nhất trên tập Validation này
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            val_pred_binary = (val_prob >= thresh).astype(int)
            score = f1_score(y_val, val_pred_binary)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
        
        oof_preds[val_idx] = val_prob
        f1_scores.append(best_f1)
        
        # Predict Test (cộng dồn rồi chia trung bình sau)
        test_preds += clf.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        
        print(f"Fold {fold+1}: Best F1 = {best_f1:.4f} at Threshold = {best_thresh:.2f}")

    print(f"\n>>> Average F1-Score (CV): {np.mean(f1_scores):.4f}")

    # 5. Xuất Feature Importance (Xem đặc trưng nào quan trọng nhất)
    lgb.plot_importance(clf, max_num_features=20, importance_type='gain', figsize=(10,6))
    plt.title("Top 20 Important Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature importance plot.")

    # 6. Tạo file Submission
    # Lấy ngưỡng trung bình của các fold (hoặc chọn thủ công dựa trên kết quả trên)
    FINAL_THRESHOLD = 0.5  # Có thể chỉnh lại, ví dụ 0.8 nếu mô hình bắt nhầm nhiều quá
    
    submission = pd.DataFrame()
    submission['object_id'] = test_df['object_id']
    submission['prediction'] = (test_preds >= FINAL_THRESHOLD).astype(int)
    
    submission.to_csv("submission.csv", index=False)
    print(">>> SUCCESS! Created 'submission.csv'. Ready to submit.")

if __name__ == "__main__":
    train_and_predict()