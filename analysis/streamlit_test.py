import streamlit as st
import pandas as pd
from pathlib import Path
from collections import defaultdict

@st.cache_data
def load_data(base_dir: str):
    base_path = Path(base_dir)
    tsv_files = list(base_path.rglob("metrics_*.tsv"))
    
    grouped_files = defaultdict(list)
    for file in tsv_files:
        core_name = file.stem.replace("metrics_", "")
        if "_small_" in core_name:
            size = "small"
            model_name, combination = core_name.split("_small_")
        elif "_normal_" in core_name:
            size = "normal"
            model_name, combination = core_name.split("_normal_")
        else:
            continue
            
        grouped_files[(size, combination)].append((model_name, file))
    return grouped_files


def process_configuration(files):
    all_models_data = []
    for model_name, file_path in files:
        df = pd.read_csv(file_path, sep="\t")
        df["model"] = model_name
        all_models_data.append(df)
        
    combo_df = pd.concat(all_models_data, ignore_index=True)
    
    # Isolate baseline data as denominator for rmsse and mase
    baseline_df = combo_df[combo_df['model'] == 'SeasonNaive'][['id', 'horizon_mae', 'horizon_rmse']]
    
    combo_df = combo_df.merge(baseline_df, on='id', suffixes=('', '_base'), how='left')        
    combo_df['mase'] = combo_df['horizon_mae'] / combo_df['horizon_mae_base'].replace(0, pd.NA)
    combo_df['rmsse'] = combo_df['horizon_rmse'] / combo_df['horizon_rmse_base'].replace(0, pd.NA)
    
    return combo_df

st.set_page_config(page_title="ZS Model Dashboard", layout="wide")
st.title("Energy Load Forecast Evaluation")

st.sidebar.header("Data Source")
metrics_dir = st.sidebar.text_input("Metrics Directory", r"C:\Users\lsb-ps\Documents\predictions_overview_lean")

if Path(metrics_dir).exists():
    grouped_files = load_data(metrics_dir)
    
    if grouped_files:
        st.sidebar.header("Configuration")
        # Extract available sizes and combinations
        available_configs = list(grouped_files.keys())
        sizes = list(set([cfg[0] for cfg in available_configs]))
        
        selected_size = st.sidebar.selectbox("Select Size", sizes)
        
        # Filter combinations based on selected size
        valid_combos = [cfg[1] for cfg in available_configs if cfg[0] == selected_size]
        selected_combo = st.sidebar.selectbox("Select Combination", valid_combos)
        
        files_to_process = grouped_files[(selected_size, selected_combo)]
        combo_df = process_configuration(files_to_process)
        total_companies = combo_df['id'].nunique()
        
        st.write(f"### Evaluating **{total_companies:,}** Companies | Size: `{selected_size.upper()}` | Combo: `{selected_combo.upper()}`")
        
        # Global Aggregations
        st.subheader("Aggregated Metrics Summary")
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
        
        st.dataframe(summary_stats.set_index('model'), use_container_width=True)



        
        # Win Rates
        st.subheader("Company-Level Win Rates")
        metrics_rules = {
            'horizon_mae': 'min',
            'horizon_rmse': 'min',
            'horizon_smape': 'min',
            'mase': 'min',
            'rmsse': 'min'
        }
        
        # Create tabs for clean visualization of different metrics
        tabs = st.tabs([m.replace('horizon_', '').upper() for m in metrics_rules.keys()])
        
        for tab, (metric, rule) in zip(tabs, metrics_rules.items()):
            with tab:
                pivot_df = combo_df.pivot_table(index='id', columns='model', values=metric, aggfunc='mean')            
                
                if rule == 'max':
                    best_models = pivot_df.idxmax(axis=1)
                elif rule == 'abs_min':
                    best_models = pivot_df.abs().idxmin(axis=1)
                else: 
                    best_models = pivot_df.idxmin(axis=1)
                    
                win_counts = best_models.value_counts().reset_index()
                win_counts.columns = ['Model', 'Wins']
                
                st.bar_chart(win_counts.set_index('Model'))
                
    else:
        st.warning("No metrics files found in the specified directory.")
else:
    st.error("Invalid directory path.")