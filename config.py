# config.py
import os

# --- ĐƯỜNG DẪN ---
# Thay đổi dấu chấm '.' thành đường dẫn thật nếu data nằm chỗ khác
DATA_ROOT = './data/' 
TRAIN_LOG_PATH = os.path.join(DATA_ROOT, 'train_log.csv')
TEST_LOG_PATH = os.path.join(DATA_ROOT, 'test_log.csv')
SUBMISSION_SAMPLE = os.path.join(DATA_ROOT, 'sample_submission.csv')

# --- HẰNG SỐ VẬT LÝ (LSST Filters - Chuẩn) ---
# Hệ số tắt dần cho từng bước sóng (R_lambda = A_lambda / E(B-V))
# Sử dụng giá trị tiêu chuẩn từ Schlafly & Finkbeiner (2011) cho bộ lọc LSST
R_LAMBDA = {
    'u': 4.239, 
    'g': 3.303, 
    'r': 2.285, 
    'i': 1.698, 
    'z': 1.263, 
    'y': 1.061
}

# Danh sách các filter để tạo cột
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# --- CẤU HÌNH MODEL ---
NUM_SPLITS = 20  # Tổng số folder split
SEED = 42