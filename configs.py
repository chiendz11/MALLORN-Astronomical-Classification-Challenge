# config.py
import os

# --- ĐƯỜNG DẪN ---
# Thay đổi dấu chấm '.' thành đường dẫn thật nếu data nằm chỗ khác
DATA_ROOT = '.' 
TRAIN_LOG_PATH = os.path.join(DATA_ROOT, 'train_log.csv')
TEST_LOG_PATH = os.path.join(DATA_ROOT, 'test_log.csv')
SUBMISSION_SAMPLE = os.path.join(DATA_ROOT, 'sample_submission.csv')

# --- HẰNG SỐ VẬT LÝ (LSST Filters) ---
# Hệ số tắt dần cho từng bước sóng (dùng để sửa độ sáng)
R_LAMBDA = {
    'u': 4.81, 
    'g': 3.64, 
    'r': 2.70, 
    'i': 2.06, 
    'z': 1.58, 
    'y': 1.31
}

# Danh sách các filter để tạo cột
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# --- CẤU HÌNH MODEL ---
NUM_SPLITS = 20  # Tổng số folder split
SEED = 42