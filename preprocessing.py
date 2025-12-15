# preprocessing.py
import os
import pandas as pd
import numpy as np
import config
from tqdm import tqdm # Thư viện tạo thanh loading bar (pip install tqdm)

def correct_flux(df, ebv_map):
    """Hiệu chỉnh độ sáng dựa trên EBV và Filter"""
    # Map EBV vào dataframe lightcurve
    df['EBV'] = df['object_id'].map(ebv_map)
    
    # Map hệ số R_lambda
    df['R_val'] = df['Filter'].map(config.R_LAMBDA)
    
    # Công thức vật lý: Flux_corr = Flux * 10^(0.4 * R * EBV)
    # Xử lý NaN cho EBV nếu có
    df['EBV'] = df['EBV'].fillna(0)
    correction_factor = np.power(10, 0.4 * df['R_val'] * df['EBV'])
    return df['Flux'] * correction_factor

def extract_features(lc_df):
    """
    Biến đổi dữ liệu chuỗi thời gian (nhiều dòng) thành đặc trưng (1 dòng)
    """
    # 1. Thống kê chung (Global features)
    # Tính trên toàn bộ các filter gộp lại
    global_aggs = lc_df.groupby('object_id')['Flux_corr'].agg(
        ['mean', 'std', 'max', 'min', 'count']
    ).add_prefix('global_')
    
    # Tính Amplitude (Biên độ dao động)
    global_aggs['global_amplitude'] = global_aggs['global_max'] - global_aggs['global_min']

    # 2. Thống kê theo từng Filter (Per-filter features)
    # Pivot table để tách các filter ra thành cột riêng
    # Ví dụ: u_mean, u_max, g_mean, g_max...
    filter_aggs = lc_df.pivot_table(
        index='object_id',
        columns='Filter',
        values='Flux_corr',
        aggfunc=['mean', 'max', 'std']
    )
    # Làm phẳng tên cột (MultiIndex -> Single Index)
    filter_aggs.columns = [f"{col[1]}_{col[0]}" for col in filter_aggs.columns]
    
    # 3. Đặc trưng màu sắc (Color Features) - Cực quan trọng cho TDE
    # Tính chênh lệch trung bình giữa các dải màu liền kề: u-g, g-r, r-i...
    # Cần đảm bảo cột tồn tại trước khi trừ
    features = pd.concat([global_aggs, filter_aggs], axis=1)
    
    pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]
    for f1, f2 in pairs:
        c1, c2 = f'{f1}_mean', f'{f2}_mean'
        if c1 in features.columns and c2 in features.columns:
            features[f'{f1}_minus_{f2}'] = features[c1] - features[c2]
            
    return features

def process_dataset(mode='train'):
    """
    Hàm chính để chạy toàn bộ quy trình xử lý dữ liệu
    mode: 'train' hoặc 'test'
    """
    print(f"\n>>> BẮT ĐẦU XỬ LÝ DỮ LIỆU: {mode.upper()} SET")
    
    # 1. Đọc file Log (Mục lục)
    log_path = config.TRAIN_LOG_PATH if mode == 'train' else config.TEST_LOG_PATH
    log_df = pd.read_csv(log_path)
    
    # Tạo từ điển EBV để tra cứu nhanh: object_id -> EBV
    ebv_map = log_df.set_index('object_id')['EBV'].to_dict()
    
    all_features = []
    
    # 2. Duyệt qua từng folder Split
    # Folder tên là split_01, split_02... đến split_20
    split_folders = sorted([f"split_{i:02d}" for i in range(1, config.NUM_SPLITS + 1)])
    
    for folder in tqdm(split_folders, desc="Processing Splits"):
        folder_path = os.path.join(config.DATA_ROOT, folder)
        file_name = f"{mode}_full_lightcurves.csv"
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: Không tìm thấy {file_path}")
            continue
            
        # Đọc dữ liệu lightcurve
        lc_df = pd.read_csv(file_path)
        
        # Chỉ giữ lại các object có trong log_df (để an toàn)
        valid_ids = set(log_df['object_id'])
        lc_df = lc_df[lc_df['object_id'].isin(valid_ids)].copy()
        
        if lc_df.empty: continue

        # --- BƯỚC QUAN TRỌNG: SỬA LỖI VẬT LÝ ---
        lc_df['Flux_corr'] = correct_flux(lc_df, ebv_map)
        
        # --- TRÍCH XUẤT ĐẶC TRƯNG ---
        # Gom nhóm 100 dòng thành 1 dòng
        batch_features = extract_features(lc_df)
        
        # Reset index để đưa object_id thành cột
        batch_features = batch_features.reset_index()
        all_features.append(batch_features)
        
    # 3. Gộp tất cả lại
    full_features_df = pd.concat(all_features, ignore_index=True)
    
    # 4. Merge với file Log để lấy nhãn (Target) và Redshift (Z)
    final_df = log_df.merge(full_features_df, on='object_id', how='left')
    
    # Lưu ra file CSV để dùng cho bước Train
    output_filename = f"processed_{mode}_data.csv"
    final_df.to_csv(output_filename, index=False)
    print(f">>> Đã lưu file xử lý xong: {output_filename} (Shape: {final_df.shape})")

if __name__ == "__main__":
    # Chạy xử lý cho cả Train và Test
    process_dataset(mode='train')
    process_dataset(mode='test')