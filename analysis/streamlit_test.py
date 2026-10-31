import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from collections import defaultdict

BASELINE_MODELS = ['HistoricAvg', 'SeasonNaive', 'TSMixerOutDom', 'CNNOutDom']

@st.cache_data
def load_data(base_dir: str):
    base_path = Path(base_dir)
    print(f"DEBUG: Searching in absolute path: {base_path.resolve()}")
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


@st.cache_data
def load_wilcoxon_stats(base_dir: str):
    base_path = Path(base_dir)
    wilcoxon_file = base_path / "master_wilcoxon_tests.parquet"
    if wilcoxon_file.exists():
        return pd.read_parquet(wilcoxon_file, engine='pyarrow')
    return pd.DataFrame()


def process_configuration(files):
    all_models_data = []
    for model_name, file_path in files:
        df = pd.read_csv(file_path, sep="\t")
        df["model"] = model_name
        all_models_data.append(df)
        
    combo_df = pd.concat(all_models_data, ignore_index=True)
    return combo_df


st.set_page_config(page_title="ZS Model Dashboard", layout="wide")
st.title("Energy Load Forecast Evaluation")

st.sidebar.header("Data Source")
metrics_dir = st.sidebar.text_input("Metrics Directory", r"C:\Users\lsb-ps\Documents\predictions_overview_lean")

if Path(metrics_dir).exists():
    grouped_files = load_data(metrics_dir)
    df_wilcoxon = load_wilcoxon_stats(metrics_dir)
    
    if grouped_files:
        st.sidebar.header("Configuration")
        
        available_configs = list(grouped_files.keys())
        sizes = list(set([cfg[0] for cfg in available_configs]))
        selected_size = st.sidebar.selectbox("Select Size", sizes)
        
        valid_combos = [cfg[1] for cfg in available_configs if cfg[0] == selected_size]
        selected_combo = st.sidebar.selectbox("Select Combination", valid_combos)
        
        files_to_process = grouped_files[(selected_size, selected_combo)]
        combo_df = process_configuration(files_to_process)
        total_companies = combo_df['id'].nunique()
        
        st.write(f"### Evaluating **{total_companies:,}** Companies | Size: `{selected_size.upper()}` | Combo: `{selected_combo.upper()}`")
        
        # --- Aggregated Metrics Summary ---
        st.subheader("Aggregated Metrics Summary")
        
        agg_functions = {
            'horizon_rmae_median': 'median',
            'horizon_rmse_median': 'median',
            'horizon_smape_median_median': 'median', # Handles the flattened multi-index
            'horizon_da_mean': 'mean'
        }
        
        valid_agg = {k: v for k, v in agg_functions.items() if k in combo_df.columns}
        
        summary_stats = combo_df.groupby('model').agg(valid_agg).reset_index()
        summary_stats.rename(columns={
            'horizon_rmae_median': 'rMAE (Median)', 
            'horizon_rmse_median': 'RMSE (Median)',
            'horizon_smape_median_median': 'sMAPE (Median)', 
            'horizon_da_mean': 'DA (Mean)'
        }, inplace=True)
        
        st.dataframe(summary_stats.set_index('model'), use_container_width=True)

        st.subheader("Metric Comparison")
        metric_cols = [col for col in summary_stats.columns if col != 'model']
        val_tabs = st.tabs(metric_cols)

        for tab, metric_col in zip(val_tabs, metric_cols):
            with tab:
                plot_df = summary_stats[['model', metric_col]]
                
                color_condition = alt.condition(
                    alt.FieldOneOfPredicate(field='model', oneOf=BASELINE_MODELS),
                    alt.value('#ff7f0e'), 
                    alt.value('#1f77b4')  
                )
                
                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('model:N', title='Model Name', sort='y' if metric_col != 'DA (Mean)' else '-y'),
                    y=alt.Y(f'{metric_col}:Q', title=f'{metric_col}'),
                    color=color_condition
                ).properties(width='container')
                
                st.altair_chart(chart, use_container_width=True)


        st.write("---")
        # --- Statistical Significance Section ---
        st.subheader("Statistical Significance (Wilcoxon Signed-Rank Test)")
        
        if not df_wilcoxon.empty:
            cohort_wilcoxon = df_wilcoxon[
                (df_wilcoxon['size_category'] == selected_size) & 
                (df_wilcoxon['combo_prefix'] == selected_combo)
            ].copy()
            
            if not cohort_wilcoxon.empty:
                cohort_wilcoxon.drop(columns=['size_category', 'combo_prefix'], inplace=True)
                
                # 1. Map backend metric names to presentable titles
                metric_name_map = {
                    'horizon_rmae_median': 'rMAE (Median)',
                    'horizon_smape_median_median': 'sMAPE (Median)',
                    'horizon_smape_median': 'sMAPE (Median)', # Fallback
                    'horizon_rmse_median': 'RMSE (Median)',
                    'horizon_da_mean': 'DA (Mean)'
                }
                cohort_wilcoxon['target_metric'] = cohort_wilcoxon['target_metric'].map(metric_name_map).fillna(cohort_wilcoxon['target_metric'])
                
                # 2. Rename columns cleanly
                cohort_wilcoxon.rename(columns={
                    'target_metric': 'Metric',
                    'model_name': 'Model',
                    'baseline_model': 'Baseline',
                    'n_valid_samples': 'Valid Samples (N)',
                    'dropped_infs_nans': 'Dropped (Inf/NaN)',
                    'median_difference': 'Median Diff.',
                    'w_statistic': 'W-Stat',
                    'p_value': 'p-value',
                    'is_significant': 'Significant',
                    'conclusion': 'Conclusion'
                }, inplace=True)
                
                # 3. Format visual display
                def color_significance(val):
                    if isinstance(val, bool):
                        return 'background-color: #2e7d32; color: white' if val else 'background-color: #c62828; color: white'
                    return ''
                    
                styled_df = cohort_wilcoxon.style.map(color_significance, subset=['Significant'])
                
                # Intercept p-values that are incredibly small to avoid showing 0.0000
                styled_df = styled_df.format({
                    'p-value': lambda x: '< 0.001' if (x < 0.001 and x >= 0) else f"{x:.4f}",
                    'Median Diff.': "{:.4f}",
                    'W-Stat': "{:.0f}"
                })
                    
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No statistical tests found for Size: {selected_size}, Combo: {selected_combo}. Run the export script.")
        else:
            st.warning("Wilcoxon parquet file not found. Ensure 'master_wilcoxon_tests.parquet' exists in the metrics directory.")


        st.write("---")
        # --- Cross-Combination Analysis ---
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
                
                rmae_val = df['horizon_rmae_median'].median() if 'horizon_rmae_median' in df.columns else pd.NA
                
                # Check for either the flattened multi-index or standard fallback
                if 'horizon_smape_median_median' in df.columns:
                    smape_val = df['horizon_smape_median_median'].median()
                elif 'horizon_smape_median' in df.columns:
                    smape_val = df['horizon_smape_median'].median()
                else:
                    smape_val = pd.NA
                
                combo_comparisons.append({
                    "Combination": combo,
                    "rMAE (Median)": rmae_val,
                    "sMAPE (Median)": smape_val
                })
                
        if combo_comparisons:
            comparison_df = pd.DataFrame(combo_comparisons).dropna().set_index("Combination")
            st.write(f"**{target_model}** Performance by Combination")
            
            tab_rmae, tab_smape = st.tabs(["rMAE (Median)", "sMAPE (Median)"])
            
            plot_df = comparison_df.reset_index()
            line_color = '#ff7f0e' if target_model in BASELINE_MODELS else '#1f77b4'
            
            def create_line_chart(df, y_col):
                # Fixed bug: Removed stray backticks inside the f-string format 
                return alt.Chart(df).mark_line(color=line_color, point=True).encode(
                    x=alt.X('Combination:N', title='Combination Horizon'),
                    y=alt.Y(f'{y_col}:Q', title=f'{y_col}')
                ).properties(width='container')
            
            with tab_rmae:
                st.altair_chart(create_line_chart(plot_df, "rMAE (Median)"), use_container_width=True)
            with tab_smape:
                st.altair_chart(create_line_chart(plot_df, "sMAPE (Median)"), use_container_width=True)


        st.write("---")
        # --- Win Rates ---
        st.subheader("Company-Level Win Rates")
        
        metrics_rules = {
            'horizon_rmae_median': 'min',
            'horizon_rmse_median': 'min',
            'horizon_smape_median_median': 'min',
            'horizon_da_mean': 'max'
        }
        
        existing_metrics = {k: v for k, v in metrics_rules.items() if k in combo_df.columns}
        
        if existing_metrics:
            tabs = st.tabs([m.replace('horizon_', '').replace('_median_median', '').replace('_', ' ').upper() for m in existing_metrics.keys()])
            
            for tab, (metric, rule) in zip(tabs, existing_metrics.items()):
                with tab:
                    clean_combo_df = combo_df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=[metric])
                    pivot_df = clean_combo_df.pivot_table(index='id', columns='model', values=metric, aggfunc='median')            
                    
                    if rule == 'max':
                        best_models = pivot_df.idxmax(axis=1)
                    else: 
                        best_models = pivot_df.idxmin(axis=1)
                        
                    win_counts = best_models.value_counts().reset_index()
                    win_counts.columns = ['Model', 'Wins']
                    
                    color_condition = alt.condition(
                        alt.FieldOneOfPredicate(field='Model', oneOf=BASELINE_MODELS),
                        alt.value('#ff7f0e'),
                        alt.value('#1f77b4')
                    )
                    
                    chart = alt.Chart(win_counts).mark_bar().encode(
                        x=alt.X('Model:N', title='Model Name', sort='-y'),
                        y=alt.Y('Wins:Q', title='Company Wins (Count)'),
                        color=color_condition
                    ).properties(width='container')
                    
                    st.altair_chart(chart, use_container_width=True)
                
    else:
        st.warning("No metrics files found in the specified directory.")
else:
    st.error("Invalid directory path.")