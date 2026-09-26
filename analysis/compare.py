import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_filename_metadata(file_stem: str):
    core_name = file_stem.replace("metrics_", "")
    
    if "_small_" in core_name:
        size = "small"
        model_name, combination = core_name.split("_small_")
    elif "_normal_" in core_name:
        size = "normal"
        model_name, combination = core_name.split("_normal_")
    else:
        raise ValueError(f"Could not determine size (small/normal) from filename: {file_stem}")
        
    return model_name, size, combination


def generate_comparison_reports(base_dir: str | Path, output_dir: str | Path):
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover and group files by (size, combination)
    tsv_files = list(base_path.rglob("metrics_*.tsv"))
    if not tsv_files:
        logging.warning(f"No metrics TSV files found in {base_path}")
        return
        
    grouped_files = defaultdict(list)
    for file in tsv_files:
        model_name, size, combination = parse_filename_metadata(file.stem)
        grouped_files[(size, combination)].append((model_name, file))
        
    logging.info(f"Found {len(tsv_files)} files across {len(grouped_files)} configurations.")

    # Win calculatino of each metric
    metrics_rules = {
        'horizon_mae': 'min',
        'horizon_rmse': 'min',
        'horizon_smape': 'min',
        'mase': 'min',
        'rmsse': 'min'
    }

    # 2. Process each configuration independently
    for (size, combination), files in grouped_files.items():
        logging.info(f"Aggregating {size} - {combination} ({len(files)} models)")
        
        all_models_data = []
        for model_name, file_path in files:
            df = pd.read_csv(file_path, sep="\t")
            df["model"] = model_name
            all_models_data.append(df)
            
        combo_df = pd.concat(all_models_data, ignore_index=True)
        total_companies = combo_df['id'].nunique()

        # Isolate baseline data as denominator for rmsse and mase
        baseline_df = combo_df[combo_df['model'] == 'SeasonNaive'][['id', 'horizon_mae', 'horizon_rmse']]
        if baseline_df.empty:
            logging.warning(f"'seasonal_naive' model not found in {size}-{combination}. MASE and RMSSE will be NaN.")

        combo_df = combo_df.merge(baseline_df, on='id', suffixes=('', '_base'), how='left')        
        # Calculate Relative Metrics, replace 0 with NaN
        combo_df['mase'] = combo_df['horizon_mae'] / combo_df['horizon_mae_base'].replace(0, pd.NA)
        combo_df['rmsse'] = combo_df['horizon_rmse'] / combo_df['horizon_rmse_base'].replace(0, pd.NA)
        
        # 3. Calculate Global Means & Medians
        agg_functions = {
            'horizon_mae': 'median',
            'horizon_rmse': 'median',
            'mase': 'median',
            'rmsse': 'median',
            'horizon_smape': 'mean',
            'horizon_da': 'mean',
            'horizon_bias': 'mean'
        }
        summary_stats = combo_df.groupby('model').agg(agg_functions).reset_index()
        summary_stats.rename(columns={
            'horizon_mae': 'MAE(median)', 
            'horizon_rmse': 'RMSE(median)',
            'mase': 'MASE(median)',
            'rmsse': 'RMSSE(median)',
            'horizon_smape': 'sMAPE(mean)', 
            'horizon_da': 'DA(mean)', 
            'horizon_bias': 'Bias(mean)'
        }, inplace=True)
        
        # 4. Calculate Company-Level Win Rates for EVERY metric
        win_results = {}
        for metric, rule in metrics_rules.items():
            pivot_df = combo_df.pivot_table(index='id', columns='model', values=metric, aggfunc='mean')            
            # Apply the specific win logic for the metric
            if rule == 'max':
                best_models = pivot_df.idxmax(axis=1)
            elif rule == 'abs_min':
                best_models = pivot_df.abs().idxmin(axis=1)
            else: # 'min'
                best_models = pivot_df.idxmin(axis=1)
                
            win_results[metric] = best_models.value_counts()

        # 5. Export to plain text report
        report_filename = out_path / f"comparison_report_{size}_{combination}.txt"
        with open(report_filename, "w") as f:
            f.write(f" MODEL COMPARISON REPORT\n")
            f.write(f" Size: {size.upper()} | Combination: {combination.upper()}\n")
            f.write(f" Total Companies Evaluated: {total_companies:,}\n")
            f.write(f"====================================================\n\n")
            
            f.write(" WIN RATES PER METRIC\n")
            for metric, wins in win_results.items():
                metric_clean = metric.replace('horizon_', '').upper()
                f.write(f"\nMetric: {metric_clean}\n")
                
                # Calculate model stats, iterate based on win rate
                model_stats = []
                for model in summary_stats['model']:
                    count = wins.get(model, 0)
                    pct = (count / total_companies) * 100
                    model_stats.append({'model': model, 'count': count, 'pct': pct})

                model_stats.sort(key=lambda x: x['count'], reverse=True)
                for stat in model_stats:
                    f.write(f"  - {stat['model']}: {stat['count']} wins ({stat['pct']:.1f}%)\n")
                
            f.write("\n\nMETRICS SUMMARY\n")
            f.write(summary_stats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            
        logging.info(f"Saved comparison report to {report_filename}")

if __name__ == "__main__":
    METRICS_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean"
    OUTPUT_REPORT_DIR = r"C:\Users\lsb-ps\Documents\predictions_overview_lean"
    
    generate_comparison_reports(METRICS_DIR, OUTPUT_REPORT_DIR)