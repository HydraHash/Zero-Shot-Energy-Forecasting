import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

BASELINE_MODELS = ['HistoricAvg', 'SeasonNaive', 'TSMixer', 'CNN']
TOP_ZS_MODELS = ['Chronos2', 'Bolt43m', 'Bolt21m', 'Moirai20']
TOP_BASELINE_MODELS = ['SeasonNaive', 'TSMixer']

MONTH_TO_SEASON = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

SEASON_ORDER = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Autumn': 4}


def process_combo_size(dataset, size: str, combo: str):
    print(f"Processing size='{size}', combo='{combo}'...")
    t0 = time.time()

    scanner = dataset.scanner(
        columns=['id', 'horizon_start', 'model_name', 'mae'],
        filter=(ds.field('combo') == combo) & (ds.field('size') == size)
    )
    tab = scanner.to_table()
    if tab.num_rows == 0:
        print(f"  Empty slice for {size} {combo}")
        return []

    df = tab.to_pandas()
    df['model_name'] = df['model_name'].replace({'kairos23m': 'Kairos23m', 'kairos50m': 'Kairos50m'})
    df['month'] = df['horizon_start'].str.slice(5, 7).astype('int8')
    df['season'] = df['month'].map(MONTH_TO_SEASON)
    df['architecture'] = np.where(df['model_name'].isin(BASELINE_MODELS), 'Baseline', 'Zero-Shot')

    # Merge SeasonNaive for rMAE
    sn = df[df['model_name'] == 'SeasonNaive'][['id', 'horizon_start', 'mae']].rename(columns={'mae': 'sn_mae'})
    if not sn.empty:
        df = df.merge(sn, on=['id', 'horizon_start'], how='left')
        df['rmae'] = df['mae'] / df['sn_mae'].replace(0, np.nan)
    else:
        df['rmae'] = np.nan

    records = []

    # 1. Per-Model Monthly Aggregations
    monthly_grouped = df.groupby(['model_name', 'architecture', 'month', 'season'])
    for (m_name, arch, m_num, s_name), group in monthly_grouped:
        maes = group['mae'].dropna()
        rmaes = group['rmae'].dropna()
        records.append({
            'size': size,
            'combo': combo,
            'model_name': m_name,
            'architecture': arch,
            'granularity': 'Monthly',
            'period_num': int(m_num),
            'period_label': MONTH_NAMES[m_num],
            'season': s_name,
            'median_mae': float(maes.median()) if len(maes) else np.nan,
            'mean_mae': float(maes.mean()) if len(maes) else np.nan,
            'p25_mae': float(maes.quantile(0.25)) if len(maes) else np.nan,
            'p75_mae': float(maes.quantile(0.75)) if len(maes) else np.nan,
            'median_rmae': float(rmaes.median()) if len(rmaes) else np.nan,
            'mean_rmae': float(rmaes.mean()) if len(rmaes) else np.nan,
            'sample_count': int(len(group))
        })

    # 2. Per-Model Seasonal Aggregations
    seasonal_grouped = df.groupby(['model_name', 'architecture', 'season'])
    for (m_name, arch, s_name), group in seasonal_grouped:
        maes = group['mae'].dropna()
        rmaes = group['rmae'].dropna()
        records.append({
            'size': size,
            'combo': combo,
            'model_name': m_name,
            'architecture': arch,
            'granularity': 'Seasonal',
            'period_num': SEASON_ORDER[s_name],
            'period_label': s_name,
            'season': s_name,
            'median_mae': float(maes.median()) if len(maes) else np.nan,
            'mean_mae': float(maes.mean()) if len(maes) else np.nan,
            'p25_mae': float(maes.quantile(0.25)) if len(maes) else np.nan,
            'p75_mae': float(maes.quantile(0.75)) if len(maes) else np.nan,
            'median_rmae': float(rmaes.median()) if len(rmaes) else np.nan,
            'mean_rmae': float(rmaes.mean()) if len(rmaes) else np.nan,
            'sample_count': int(len(group))
        })

    # 3. Macro Aggregates across Architectures:
    # (a) All Zero-Shot, (b) All Baseline, (c) Top Zero-Shot, (d) Top Baseline
    macro_configs = [
        ('Zero-Shot Aggregate', 'Zero-Shot', df['architecture'] == 'Zero-Shot'),
        ('Baseline Aggregate', 'Baseline', df['architecture'] == 'Baseline'),
        ('Top Zero-Shot (Chronos/Bolt/Moirai)', 'Zero-Shot', df['model_name'].isin(TOP_ZS_MODELS)),
        ('Top Baseline (SeasonNaive/TSMixer)', 'Baseline', df['model_name'].isin(TOP_BASELINE_MODELS)),
    ]

    for macro_name, macro_arch, mask in macro_configs:
        sub = df[mask]
        if sub.empty:
            continue

        # Monthly
        for m_num, m_group in sub.groupby('month'):
            maes = m_group['mae'].dropna()
            rmaes = m_group['rmae'].dropna()
            s_name = MONTH_TO_SEASON[m_num]
            records.append({
                'size': size,
                'combo': combo,
                'model_name': macro_name,
                'architecture': macro_arch,
                'granularity': 'Monthly',
                'period_num': int(m_num),
                'period_label': MONTH_NAMES[m_num],
                'season': s_name,
                'median_mae': float(maes.median()) if len(maes) else np.nan,
                'mean_mae': float(maes.mean()) if len(maes) else np.nan,
                'p25_mae': float(maes.quantile(0.25)) if len(maes) else np.nan,
                'p75_mae': float(maes.quantile(0.75)) if len(maes) else np.nan,
                'median_rmae': float(rmaes.median()) if len(rmaes) else np.nan,
                'mean_rmae': float(rmaes.mean()) if len(rmaes) else np.nan,
                'sample_count': int(len(m_group))
            })

        # Seasonal
        for s_name, s_group in sub.groupby('season'):
            maes = s_group['mae'].dropna()
            rmaes = s_group['rmae'].dropna()
            records.append({
                'size': size,
                'combo': combo,
                'model_name': macro_name,
                'architecture': macro_arch,
                'granularity': 'Seasonal',
                'period_num': SEASON_ORDER[s_name],
                'period_label': s_name,
                'season': s_name,
                'median_mae': float(maes.median()) if len(maes) else np.nan,
                'mean_mae': float(maes.mean()) if len(maes) else np.nan,
                'p25_mae': float(maes.quantile(0.25)) if len(maes) else np.nan,
                'p75_mae': float(maes.quantile(0.75)) if len(maes) else np.nan,
                'median_rmae': float(rmaes.median()) if len(rmaes) else np.nan,
                'mean_rmae': float(rmaes.mean()) if len(rmaes) else np.nan,
                'sample_count': int(len(s_group))
            })

    elapsed = round(time.time() - t0, 2)
    print(f"  Finished {size} {combo} in {elapsed}s, generated {len(records)} aggregate rows")
    return records


def main():
    raw_path = Path(r"C:\Users\lsb-ps\Documents\predictions_overview_lean\all_models_raw_mae.parquet")
    out_dir = Path(r"C:\Users\lsb-ps\Documents\predictions_overview_lean")
    target_file = out_dir / "seasonal_monthly_metrics.parquet"

    print(f"Opening raw dataset at {raw_path}...")
    dataset = ds.dataset(str(raw_path), format="parquet")

    all_records = []
    combos = ['c1', 'c2', 'c3', 'c4']
    sizes = ['small', 'normal']

    start_total = time.time()
    for s in sizes:
        for c in combos:
            res = process_combo_size(dataset, s, c)
            all_records.extend(res)

    print(f"\nAll chunks processed in {round(time.time() - start_total, 2)}s. Total records: {len(all_records)}")
    df_out = pd.DataFrame(all_records)

    print(f"Saving to {target_file}...")
    df_out.to_parquet(target_file, engine='pyarrow', index=False)
    print(f"File size: {round(target_file.stat().st_size / 1024, 2)} KB")

    # Also copy to analysis directory for easy local access
    analysis_copy = Path(r"C:\Users\lsb-ps\Documents\zs_implementation\analysis\seasonal_monthly_metrics.parquet")
    df_out.to_parquet(analysis_copy, engine='pyarrow', index=False)
    print(f"Copied to {analysis_copy}")


if __name__ == "__main__":
    main()

