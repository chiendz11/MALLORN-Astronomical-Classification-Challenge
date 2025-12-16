# preprocessing.py (V3 - TDE Optimized)
import os
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
from scipy.stats import skew

EPS = 1e-6


def correct_flux(df, ebv_map):
    """De-redden flux using EBV"""
    df['EBV'] = df['object_id'].map(ebv_map).fillna(0)
    df['R_val'] = df['Filter'].map(config.R_LAMBDA)
    correction = np.power(10, 0.4 * df['R_val'] * df['EBV'])
    return df['Flux'] * correction


def extract_time_features(df):
    """Time-domain features per object (TDE-critical)"""
    feats = {}

    time = df['Time (MJD)'].values
    flux = df['Flux_corr'].values

    if len(time) < 3:
        return {}

    # Peak
    peak_idx = np.argmax(flux)
    t_peak = time[peak_idx]

    feats['t_rise'] = t_peak - time.min()
    feats['t_decay'] = time.max() - t_peak

    # Early / Late flux ratio
    q25 = np.quantile(time, 0.25)
    q75 = np.quantile(time, 0.75)

    early_flux = flux[time <= q25]
    late_flux = flux[time >= q75]

    feats['early_late_flux_ratio'] = (
        np.mean(early_flux) / (np.mean(late_flux) + EPS)
        if len(early_flux) and len(late_flux) else 0
    )

    # Number of peak points (proxy for multi-peak AGN)
    feats['n_peak_points'] = np.sum(flux > 0.8 * flux.max())

    return feats


def extract_features(lc_df, log_df):
    """
    Feature Set V3:
    - Shape
    - Color
    - Time evolution
    - SNR
    - AGN suppression
    """

    features = []

    for oid, df in lc_df.groupby('object_id'):
        obj = {'object_id': oid}

        z = log_df.loc[oid, 'Z'] if oid in log_df.index else 0.0
        time_rest = df['Time (MJD)'] / (1 + z)

        df = df.copy()
        df['Time_rest'] = time_rest

        flux = df['Flux_corr'].values
        ferr = df['Flux_err'].values

        # --------------------
        # Global statistics
        # --------------------
        obj['flux_mean'] = np.mean(flux)
        obj['flux_std'] = np.std(flux)
        obj['flux_median'] = np.median(flux)
        obj['flux_max'] = np.max(flux)
        obj['flux_min'] = np.min(flux)
        obj['flux_skewness'] = skew(flux)
        obj['flux_amplitude'] = obj['flux_max'] - obj['flux_min']
        obj['burst_ratio'] = obj['flux_max'] / (abs(obj['flux_median']) + EPS)

        # Robust
        obj['flux_mad'] = np.median(np.abs(flux - obj['flux_median']))

        # Negative flux (AGN killer)
        obj['neg_flux_ratio'] = np.mean(flux < 0)

        # --------------------
        # SNR features
        # --------------------
        snr = flux / (ferr + EPS)
        obj['snr_mean'] = np.mean(snr)
        obj['snr_max'] = np.max(snr)

        # Weighted mean
        weights = 1 / (ferr ** 2 + EPS)
        obj['flux_weighted_mean'] = np.sum(flux * weights) / np.sum(weights)

        # --------------------
        # Time-domain features
        # --------------------
        time_feats = extract_time_features(df)
        obj.update(time_feats)

        # Rest-frame timescale
        obj['t_rise_rest'] = obj.get('t_rise', 0) / (1 + z)
        obj['t_decay_rest'] = obj.get('t_decay', 0) / (1 + z)

        # --------------------
        # Per-filter features
        # --------------------
        for f in ['u', 'g', 'r', 'i', 'z', 'y']:
            fdf = df[df['Filter'] == f]
            if fdf.empty:
                continue

            fflux = fdf['Flux_corr'].values
            obj[f'{f}_mean'] = np.mean(fflux)
            obj[f'{f}_max'] = np.max(fflux)
            obj[f'{f}_skewness'] = skew(fflux)

        # --------------------
        # Color features (peak & mean)
        # --------------------
        pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'y')]
        for f1, f2 in pairs:
            if f'{f1}_mean' in obj and f'{f2}_mean' in obj:
                obj[f'{f1}_minus_{f2}_mean'] = obj[f'{f1}_mean'] - obj[f'{f2}_mean']
            if f'{f1}_max' in obj and f'{f2}_max' in obj:
                obj[f'{f1}_minus_{f2}_max'] = obj[f'{f1}_max'] - obj[f'{f2}_max']

        # --------------------
        # Meta
        # --------------------
        obj['n_obs'] = len(df)
        obj['n_filters'] = df['Filter'].nunique()

        features.append(obj)

    return pd.DataFrame(features)


def process_dataset(mode='train'):
    print(f"\n>>> PROCESSING DATASET V3: {mode.upper()}")

    log_path = config.TRAIN_LOG_PATH if mode == 'train' else config.TEST_LOG_PATH
    log_df = pd.read_csv(log_path).set_index('object_id')

    ebv_map = log_df['EBV'].to_dict()
    all_features = []

    split_folders = sorted([f"split_{i:02d}" for i in range(1, config.NUM_SPLITS + 1)])

    for folder in tqdm(split_folders, desc="Processing splits"):
        file_path = os.path.join(
            config.DATA_ROOT, folder, f"{mode}_full_lightcurves.csv"
        )
        if not os.path.exists(file_path):
            continue

        lc_df = pd.read_csv(file_path)
        lc_df = lc_df[lc_df['object_id'].isin(log_df.index)]
        if lc_df.empty:
            continue

        lc_df['Flux_corr'] = correct_flux(lc_df, ebv_map)
        feats = extract_features(lc_df, log_df)

        all_features.append(feats)

    final_features = pd.concat(all_features, ignore_index=True)
    final_df = log_df.reset_index().merge(final_features, on='object_id', how='left')

    out = f"processed_{mode}_data.csv"
    final_df.to_csv(out, index=False)
    print(f">>> Saved: {out}")


if __name__ == "__main__":
    process_dataset('train')
    process_dataset('test')
