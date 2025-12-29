import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# 1. CONFIGURATION
# ==================================================================================
class Config:
    DATA_ROOT = 'data'
    TRAIN_LOG_PATH = os.path.join(DATA_ROOT, 'train_log.csv')
    TEST_LOG_PATH = os.path.join(DATA_ROOT, 'test_log.csv')
    R_LAMBDA = {'u': 4.74, 'g': 3.20, 'r': 2.27, 'i': 1.63, 'z': 1.26, 'y': 1.08}
    EPS = 1e-6
    SEEDS = [42, 2024, 777, 101, 88]
    N_FOLDS = 5

config = Config()

# ==================================================================================
# 2. FEATURE EXTRACTION (V61: V42 FULL + V60 HALF-LIFE)
# ==================================================================================

def correct_flux(df, ebv_map):
    df = df.copy()
    df['EBV'] = df['object_id'].map(ebv_map).fillna(0)
    df['R_val'] = df['Filter'].map(config.R_LAMBDA)
    correction = np.power(10, 0.4 * df['R_val'] * df['EBV'])
    return df['Flux'] * correction, df['Flux_err'] * correction

def fit_decay_shapes(time, flux):
    # V34 Logic - Proven useful in V42
    if len(time) < 3: return 0, 1.0
    t_start = np.min(time)
    t_norm = time - t_start + 1.0
    
    valid_mask = flux > 0
    if np.sum(valid_mask) < 3: return 0, 1.0
    
    t_clean = t_norm[valid_mask]
    f_clean = flux[valid_mask]
    
    y_log = np.log(f_clean)
    x_linear = t_clean.reshape(-1, 1)
    x_log = np.log(t_clean).reshape(-1, 1)
    
    try:
        model_exp = LinearRegression().fit(x_linear, y_log)
        pred_exp = model_exp.predict(x_linear)
        mse_exp = np.mean((y_log - pred_exp)**2)
        
        model_pow = LinearRegression().fit(x_log, y_log)
        pred_pow = model_pow.predict(x_log)
        mse_pow = np.mean((y_log - pred_pow)**2)
        alpha = model_pow.coef_[0]
        
        ratio = mse_pow / (mse_exp + 1e-9)
        return alpha, ratio
    except:
        return 0, 1.0

def calc_fwhm_width(time, flux, z):
    if len(flux) < 3: return 0
    max_flux = np.max(flux)
    half_max = max_flux / 2.0
    mask_above = flux >= half_max
    if np.sum(mask_above) < 2: return 0
    t_above = time[mask_above]
    width = np.max(t_above) - np.min(t_above)
    return width / (1 + z)

def calc_half_life(time, flux, t_peak, f_peak):
    # V60 Addition: Time to decay to 50%
    if len(flux) < 3: return 0
    half_flux = f_peak / 2.0
    mask_decay = (time > t_peak) & (flux <= half_flux)
    
    if np.any(mask_decay):
        t_half_cross = np.min(time[mask_decay])
        return t_half_cross - t_peak
    else:
        # If it doesn't cross, take the full duration of observation post-peak
        mask_post = time > t_peak
        if np.any(mask_post):
            return np.max(time[mask_post]) - t_peak
        return 0

def extract_features(lc_df, log_df):
    """
    V61: The Reunion.
    - Base: V42 Full (Integrals, Decay, Energy Ratio, Color Evol).
    - Add: V60 Half-Life.
    """
    features = []
    z_map = log_df['Z'].to_dict()
    grouped = lc_df.groupby('object_id')
    
    print(f"   -> Extracting V61 Features (V42 Base + Half-Life) for {len(grouped)} objects...")
    
    for oid, df in grouped:
        obj = {'object_id': oid}
        z = z_map.get(oid, 0.0)
        obj['z'] = z
        
        flux = df['Flux_corr'].values
        ferr = df['Flux_err_corr'].values 
        time = df['Time (MJD)'].values
        filters = df['Filter'].values
        
        # --- 1. GLOBAL & STABILITY (V42 Base) ---
        snr = flux / (ferr + config.EPS)
        obj['n_high_snr'] = np.sum(snr > 5)
        obj['n_detections'] = len(flux)
        obj['flux_amp'] = np.percentile(flux, 95) - np.percentile(flux, 5)
        obj['flux_skew'] = skew(flux)
        
        weights = 1.0 / (ferr**2 + config.EPS)
        w_mean = np.average(flux, weights=weights)
        obj['w_cv'] = np.sqrt(np.average((flux-w_mean)**2, weights=weights)) / (np.abs(w_mean) + config.EPS)
        obj['snr_min'] = np.min(snr) # Crucial V42 feature
        obj['snr_max'] = np.max(snr)

        # --- 2. BAND-WISE PHYSICS ---
        band_data = {} 
        
        for b in ['u', 'g', 'r', 'i', 'z', 'y']:
            mask = filters == b
            if np.any(mask):
                b_flux = flux[mask]
                b_time = time[mask]
                
                sort_idx = np.argsort(b_time)
                b_time = b_time[sort_idx]
                b_flux = b_flux[sort_idx]
                
                b_max = np.max(b_flux)
                idx_peak = np.argmax(b_flux)
                t_peak = b_time[idx_peak]
                f_peak = b_flux[idx_peak]
                
                band_data[b] = {
                    'max': b_max,
                    'flux_peak': f_peak,
                    'time_peak': t_peak,
                    'flux_tail': np.mean(b_flux[-3:]) if len(b_flux) >=3 else b_flux[-1]
                }
                
                # Standard V42
                obj[f'{b}_max'] = b_max
                obj[f'{b}_snr_max'] = np.max(b_flux / (ferr[mask][sort_idx] + config.EPS))
                
                # Integral
                if len(b_flux) > 1:
                    dt = np.diff(b_time)
                    df_avg = (b_flux[1:] + b_flux[:-1]) / 2
                    integral = np.sum(df_avg * dt)
                    obj[f'{b}_integral'] = integral / (1+z)
                    
                    # Energy Balance (V42)
                    mask_pre = b_time <= t_peak
                    mask_post = b_time > t_peak
                    E_pre = np.sum(b_flux[mask_pre]) if np.sum(mask_pre) > 0 else 0
                    E_post = np.sum(b_flux[mask_post]) if np.sum(mask_post) > 0 else 0
                    
                    if E_post > 0:
                        obj[f'{b}_energy_ratio'] = E_pre / E_post
                    else:
                        obj[f'{b}_energy_ratio'] = 10.0
                else:
                    obj[f'{b}_integral'] = 0
                    obj[f'{b}_energy_ratio'] = 0

                # Rise Time
                if idx_peak > 0:
                    obj[f'{b}_rise_time'] = (t_peak - np.min(b_time)) / (1+z)
                else:
                    obj[f'{b}_rise_time'] = 0

                # FWHM & Skew
                obj[f'{b}_fwhm'] = calc_fwhm_width(b_time, b_flux, z)
                obj[f'{b}_skew'] = skew(b_flux) if len(b_flux)>2 else 0
                
                # Lum Peak
                if b in ['u', 'g', 'r']:
                    obj[f'{b}_lum_peak'] = b_max * (z**2 + 0.001)

                # Decay Fit (V34/V42)
                if b in ['g', 'r']:
                    if len(b_flux) >= 5 and np.sum(b_time > t_peak) >= 3:
                        mask_decay = b_time > t_peak
                        alpha, ratio = fit_decay_shapes(b_time[mask_decay], b_flux[mask_decay])
                        obj[f'{b}_decay_alpha'] = alpha
                        obj[f'{b}_decay_ratio'] = ratio
                    else:
                        obj[f'{b}_decay_alpha'] = 0; obj[f'{b}_decay_ratio'] = 1.0
                
                # NEW V60: Half-Life (Only for g, r, i)
                if b in ['g', 'r', 'i']:
                    t_hl = calc_half_life(b_time, b_flux, t_peak, f_peak)
                    obj[f'{b}_half_life'] = t_hl / (1+z)

            else:
                band_data[b] = None
                obj[f'{b}_max'] = -999
                obj[f'{b}_snr_max'] = 0
                obj[f'{b}_rise_time'] = -1
                obj[f'{b}_fwhm'] = 0
                obj[f'{b}_integral'] = 0
                obj[f'{b}_energy_ratio'] = 0
                if b in ['u', 'g', 'r']: obj[f'{b}_lum_peak'] = 0; obj[f'{b}_skew'] = 0
                if b in ['g', 'r']: obj[f'{b}_decay_alpha'] = 0; obj[f'{b}_decay_ratio'] = 1.0
                if b in ['g', 'r', 'i']: obj[f'{b}_half_life'] = 0

        # --- 3. COLOR PHYSICS (V42) ---
        g_max = band_data['g']['max'] if band_data['g'] else -999
        r_max = band_data['r']['max'] if band_data['r'] else -999
        u_max = band_data['u']['max'] if band_data['u'] else -999
        
        if g_max != -999 and r_max != -999:
            obj['color_ratio_g_r'] = g_max / (r_max + config.EPS)
        else:
            obj['color_ratio_g_r'] = 0
            
        if u_max != -999 and r_max != -999:
            obj['flux_ratio_u_r'] = u_max / (r_max + config.EPS)
        else:
            obj['flux_ratio_u_r'] = 0
            
        # Color Evolution (V42)
        if band_data['g'] and band_data['r']:
            ratio_peak = band_data['g']['flux_peak'] / (band_data['r']['flux_peak'] + config.EPS)
            ratio_tail = band_data['g']['flux_tail'] / (band_data['r']['flux_tail'] + config.EPS)
            
            if ratio_peak > config.EPS:
                obj['color_evol_g_r'] = ratio_tail / ratio_peak
            else:
                obj['color_evol_g_r'] = 1.0
        else:
            obj['color_evol_g_r'] = 1.0

        features.append(obj)
        
    return pd.DataFrame(features)

# ==================================================================================
# 3. PROCESSING
# ==================================================================================
def process_data(mode='train'):
    print(f"\n>>> [1/3] PROCESSING DATA: {mode.upper()}")
    save_path = os.path.join(config.DATA_ROOT, f'processed_{mode}_v61_reunion.csv')
    
    # if os.path.exists(save_path): return pd.read_csv(save_path)

    log_path = config.TRAIN_LOG_PATH if mode == 'train' else config.TEST_LOG_PATH
    log_df = pd.read_csv(log_path).drop_duplicates(subset=['object_id']).set_index('object_id')
    ebv_map = log_df['EBV'].to_dict()
    
    lc_list = []
    split_folders = sorted(glob.glob(os.path.join(config.DATA_ROOT, 'split_*')))
    for folder in split_folders:
        fpath = os.path.join(folder, f"{mode}_full_lightcurves.csv")
        if os.path.exists(fpath):
            chunk = pd.read_csv(fpath)
            chunk = chunk[chunk['object_id'].isin(log_df.index)]
            lc_list.append(chunk)
            
    if not lc_list: return None
    full_lc = pd.concat(lc_list, ignore_index=True)
    
    f_corr, ferr_corr = correct_flux(full_lc, ebv_map)
    full_lc['Flux_corr'] = f_corr
    full_lc['Flux_err_corr'] = ferr_corr
    
    feats_df = extract_features(full_lc, log_df)
    
    out_df = log_df.reset_index().merge(feats_df, on='object_id', how='left')
    out_df = out_df.fillna(0).replace([np.inf, -np.inf], 0)
    
    out_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved V61 data: {save_path} | Shape: {out_df.shape}")
    return out_df


if __name__ == "__main__":
    process_data('train')
    process_data('test')