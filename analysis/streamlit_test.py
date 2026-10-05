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


def process_configuration(files, baseline_name):
    all_models_data = []
    for model_name, file_path in files:
        df = pd.read_csv(file_path, sep="\t")
        df["model"] = model_name
        all_models_data.append(df)
        
    combo_df = pd.concat(all_models_data, ignore_index=True)
    
    baseline_df = combo_df[combo_df['model'] == baseline_name][['id', 'horizon_mae', 'horizon_rmse']]
    
    if baseline_df.empty:
        st.warning(f"Baseline model '{baseline_name}' not found in this configuration! Relative metrics will be NaN.")
        
    combo_df = combo_df.merge(baseline_df, on='id', suffixes=('', '_base'), how='left')        
    combo_df['mase'] = combo_df['horizon_mae'] / combo_df['horizon_mae_base'].replace(0, pd.NA)
    combo_df['rmsse'] = combo_df['horizon_rmse'] / combo_df['horizon_rmse_base'].replace(0, pd.NA)
    
    return combo_df


st.set_page_config(page_title="ZS Model Dashboard", layout="wide")
st.title("Energy Load Forecast Evaluation")

st.sidebar.header("Data Source")
metrics_dir = st.sidebar.text_input("Metrics Directory", r"H:\predictions_overview_lean")

if Path(metrics_dir).exists():
    grouped_files = load_data(metrics_dir)
    
    if grouped_files:
        st.sidebar.header("Configuration")
        
        available_configs = list(grouped_files.keys())
        sizes = list(set([cfg[0] for cfg in available_configs]))
        selected_size = st.sidebar.selectbox("Select Size", sizes)
        
        valid_combos = [cfg[1] for cfg in available_configs if cfg[0] == selected_size]
        selected_combo = st.sidebar.selectbox("Select Combination", valid_combos)
        
        files_to_process = grouped_files[(selected_size, selected_combo)]
        
        available_models = list(set([m for m, f in files_to_process]))
        default_idx = available_models.index('seasonal_naive') if 'seasonal_naive' in available_models else 0
        baseline_selection = st.sidebar.selectbox("Select Baseline for MASE/RMSSE", available_models, index=default_idx)
        
        combo_df = process_configuration(files_to_process, baseline_selection)
        total_companies = combo_df['id'].nunique()
        
        st.write(f"### Evaluating **{total_companies:,}** Companies | Size: `{selected_size.upper()}` | Combo: `{selected_combo.upper()}`")
        
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

        st.subheader("Absolute Value Comparison")
        metric_cols = [col for col in summary_stats.columns if col != 'model']
        val_tabs = st.tabs(metric_cols)

        for tab, metric_col in zip(val_tabs, metric_cols):
            with tab:
                chart_data = summary_stats[['model', metric_col]].set_index('model')
                st.bar_chart(chart_data)

        st.write("---")
        st.subheader("Cross-Combination Model Analysis")
        
        all_models = set()
        for combo in valid_combos:
            for model_name, _ in grouped_files[(selected_size, combo)]:
                all_models.add(model_name)
                
        target_model = st.selectbox("Select Model to Analyze Across Combinations", list(all_models))
        
        combo_comparisons = []
        for combo in valid_combos:
            files = grouped_files[(selected_size, combo)]
            
            model_file = [f for m, f in files if m == target_model]
            if model_file:
                df = pd.read_csv(model_file[0], sep="\t")
                
                combo_comparisons.append({
                    "Combination": combo,
                    "MAE (Median)": df['horizon_mae'].median(),
                    "RMSE (Median)": df['horizon_rmse'].median(),
                    "sMAPE (Mean)": df['horizon_smape'].mean()
                })
                
        if combo_comparisons:
            comparison_df = pd.DataFrame(combo_comparisons).set_index("Combination")
            st.write(f"**{target_model}** Performance by Combination")
            
            tab_mae, tab_rmse, tab_smape = st.tabs(["MAE", "RMSE", "sMAPE"])
            
            with tab_mae:
                st.line_chart(comparison_df[["MAE (Median)"]])
            with tab_rmse:
                st.line_chart(comparison_df[["RMSE (Median)"]])
            with tab_smape:
                st.line_chart(comparison_df[["sMAPE (Mean)"]])


        st.write("---")
        st.subheader("Model Performance Change Between Combinations")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            default_a = valid_combos.index('c1') if 'c1' in valid_combos else 0
            combo_a = st.selectbox("Base Combination", valid_combos, index=default_a)
        with col2:
            default_b = valid_combos.index('c3') if 'c3' in valid_combos else (1 if len(valid_combos) > 1 else 0)
            combo_b = st.selectbox("Target Combination", valid_combos, index=default_b)
        with col3:
            change_metrics = {
                'sMAPE (Mean)': ('horizon_smape', 'mean'),
                'MAE (Median)': ('horizon_mae', 'median'),
                'RMSE (Median)': ('horizon_rmse', 'median'),
                'MASE (Median)': ('mase', 'median'),
                'RMSSE (Median)': ('rmsse', 'median'),
                'DA (Mean)': ('horizon_da', 'mean'),
                'Bias (Mean)': ('horizon_bias', 'mean')
            }
            selected_change_metric = st.selectbox("Select Metric", list(change_metrics.keys()))

        if combo_a != combo_b:
            metric_col, agg_func = change_metrics[selected_change_metric]
            
            # Process both combinations through the baseline normalization function
            df_a = process_configuration(grouped_files[(selected_size, combo_a)], baseline_selection)
            df_b = process_configuration(grouped_files[(selected_size, combo_b)], baseline_selection)
            
            # Aggregate the chosen metric for each model
            agg_a = df_a.groupby('model')[metric_col].agg(agg_func).reset_index().rename(columns={metric_col: 'Base'})
            agg_b = df_b.groupby('model')[metric_col].agg(agg_func).reset_index().rename(columns={metric_col: 'Target'})
            
            # Merge and calculate the percentage change
            merged_change = agg_a.merge(agg_b, on='model')
            merged_change['% Change'] = ((merged_change['Target'] - merged_change['Base']) / merged_change['Base']) * 100
            
            st.write(f"**Percentage Change from {combo_a.upper()} to {combo_b.upper()}**")
            st.bar_chart(merged_change[['model', '% Change']].set_index('model'))
        else:
            st.info("Please select two distinct combinations to compare.")

        
        st.subheader("Company-Level Win Rates")
        metrics_rules = {
            'horizon_mae': 'min',
            'horizon_rmse': 'min',
            'horizon_smape': 'min',
            'mase': 'min',
            'rmsse': 'min'
        }
        
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