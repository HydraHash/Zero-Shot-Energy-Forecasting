import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from collections import defaultdict

BASELINE_MODELS = ['HistoricAvg', 'SeasonNaive', 'TSMixer', 'CNN']

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
    combo_df = combo_df.replace([float('inf'), float('-inf')], pd.NA)
    return combo_df


def apply_thesis_theme(chart):
    """Applies a strict white background and black typography to any Altair chart."""
    return chart.configure(
        background='white'
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='#e5e5e5',
        domainColor='black',
        tickColor='black',
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        color='black',
        fontSize=16,
        anchor='middle'
    ).configure_legend(
        titleColor='black',
        labelColor='black'
    )


st.set_page_config(page_title="ZS Model Dashboard", layout="wide")
st.title("Energy Load Forecast Evaluation")

st.sidebar.header("Data Source")
metrics_dir = st.sidebar.text_input("Metrics Directory", r"C:\Users\lsb-ps\Documents\predictions_overview_lean")

if Path(metrics_dir).exists():
    grouped_files = load_data(metrics_dir)
    df_wilcoxon = load_wilcoxon_stats(metrics_dir)
    
    if grouped_files:
        available_configs = list(grouped_files.keys())
        all_sizes = list(set([cfg[0] for cfg in available_configs]))
        
        st.markdown("---")
        # --- Aggregated Metrics Summary ---
        st.subheader("Aggregated Metrics Summary & Comparison")
        
        agg_col1, agg_col2 = st.columns(2)
        size_agg = agg_col1.selectbox("Select Size (Aggregated)", all_sizes, key="size_agg")
        valid_combos_agg = [cfg[1] for cfg in available_configs if cfg[0] == size_agg]
        combo_agg = agg_col2.selectbox("Select Combination (Aggregated)", valid_combos_agg, key="combo_agg")
        
        agg_files = grouped_files[(size_agg, combo_agg)]
        agg_df = process_configuration(agg_files)
        total_companies_agg = agg_df['id'].nunique()
        
        st.write(f"**Evaluating {total_companies_agg:,} Companies**")
        
        agg_functions = {
            'horizon_rmae_median': 'median',
            'horizon_rmse_median': 'median',
            'horizon_smape_median_median': 'median', 
            'horizon_da_mean': 'mean'
        }
        
        valid_agg = {k: v for k, v in agg_functions.items() if k in agg_df.columns}
        
        summary_stats = agg_df.groupby('model').agg(valid_agg).reset_index()
        summary_stats.rename(columns={
            'horizon_rmae_median': 'rMAE (Median)', 
            'horizon_rmse_median': 'RMSE (Median)',
            'horizon_smape_median_median': 'sMAPE (Median)', 
            'horizon_da_mean': 'DA (Mean)'
        }, inplace=True)
        
        st.dataframe(summary_stats.set_index('model'), use_container_width=True)

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
                ).properties(width='container', height=400, title=f"Aggregated {metric_col}")
                
                st.altair_chart(apply_thesis_theme(chart), use_container_width=True)


        st.markdown("---")
        # --- Statistical Significance Section ---
        st.subheader("Statistical Significance (Wilcoxon Signed-Rank Test)")
        
        stat_col1, stat_col2 = st.columns(2)
        size_stat = stat_col1.selectbox("Select Size (Stats)", all_sizes, key="size_stat")
        valid_combos_stat = [cfg[1] for cfg in available_configs if cfg[0] == size_stat]
        combo_stat = stat_col2.selectbox("Select Combination (Stats)", valid_combos_stat, key="combo_stat")
        
        if not df_wilcoxon.empty:
            cohort_wilcoxon = df_wilcoxon[
                (df_wilcoxon['size_category'] == size_stat) & 
                (df_wilcoxon['combo_prefix'] == combo_stat)
            ].copy()
            
            if not cohort_wilcoxon.empty:
                cohort_wilcoxon.drop(columns=['size_category', 'combo_prefix'], inplace=True)
                
                metric_name_map = {
                    'horizon_rmae_median': 'rMAE (Median)',
                    'horizon_smape_median_median': 'sMAPE (Median)',
                    'horizon_smape_median': 'sMAPE (Median)', 
                    'horizon_rmse_median': 'RMSE (Median)',
                    'horizon_da_mean': 'DA (Mean)'
                }
                cohort_wilcoxon['target_metric'] = cohort_wilcoxon['target_metric'].map(metric_name_map).fillna(cohort_wilcoxon['target_metric'])
                
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
                
                def color_significance(val):
                    if isinstance(val, bool):
                        return 'background-color: #2e7d32; color: white' if val else 'background-color: #c62828; color: white'
                    return ''
                    
                styled_df = cohort_wilcoxon.style.map(color_significance, subset=['Significant'])
                
                styled_df = styled_df.format({
                    'p-value': lambda x: '< 0.001' if (x < 0.001 and x >= 0) else f"{x:.4f}",
                    'Median Diff.': "{:.4f}",
                    'W-Stat': "{:.0f}"
                })
                    
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No statistical tests found for Size: {size_stat}, Combo: {combo_stat}. Run the export script.")
        else:
            st.warning("Wilcoxon parquet file not found. Ensure 'master_wilcoxon_tests.parquet' exists in the metrics directory.")


        st.markdown("---")
        # --- Cross-Combination Analysis ---
        st.subheader("Cross-Combination Model Analysis")
        
        cross_col1, cross_col2 = st.columns(2)
        size_cross = cross_col1.selectbox("Select Size (Cross-Combo)", all_sizes, key="size_cross")
        valid_combos_cross = [cfg[1] for cfg in available_configs if cfg[0] == size_cross]
        
        all_models = set()
        for combo in valid_combos_cross:
            for model_name, _ in grouped_files[(size_cross, combo)]:
                all_models.add(model_name)
                
        target_model = cross_col2.selectbox("Select Model to Analyze Across Combinations", list(all_models), key="model_cross")
        
        combo_comparisons = []
        for combo in valid_combos_cross:
            files = grouped_files[(size_cross, combo)]
            model_file = [f for m, f in files if m == target_model]
            
            if model_file:
                df = pd.read_csv(model_file[0], sep="\t").replace([float('inf'), float('-inf')], pd.NA)
                
                rmae_val = df['horizon_rmae_median'].median() if 'horizon_rmae_median' in df.columns else pd.NA
                
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
            
            tab_rmae, tab_smape = st.tabs(["rMAE (Median)", "sMAPE (Median)"])
            
            plot_df = comparison_df.reset_index()
            line_color = '#ff7f0e' if target_model in BASELINE_MODELS else '#1f77b4'
            
            def create_line_chart(df, y_col):
                chart = alt.Chart(df).mark_line(color=line_color, point=True).encode(
                    x=alt.X('Combination:N', title='Combination Horizon'),
                    y=alt.Y(f'{y_col}:Q', title=f'{y_col}')
                ).properties(width='container', height=400, title=f"{target_model} - {y_col} Across Horizons")
                return apply_thesis_theme(chart)
            
            with tab_rmae:
                st.altair_chart(create_line_chart(plot_df, "rMAE (Median)"), use_container_width=True)
            with tab_smape:
                st.altair_chart(create_line_chart(plot_df, "sMAPE (Median)"), use_container_width=True)


        st.markdown("---")
        # --- Win Rates ---
        st.subheader("Company-Level Win Rates")
        
        win_col1, win_col2 = st.columns(2)
        size_win = win_col1.selectbox("Select Size (Win Rates)", all_sizes, key="size_win")
        valid_combos_win = [cfg[1] for cfg in available_configs if cfg[0] == size_win]
        combo_win = win_col2.selectbox("Select Combination (Win Rates)", valid_combos_win, key="combo_win")
        
        win_files = grouped_files[(size_win, combo_win)]
        win_df = process_configuration(win_files)
        
        metrics_rules = {
            'horizon_rmae_median': 'min',
            'horizon_rmse_median': 'min',
            'horizon_smape_median_median': 'min',
            'horizon_da_mean': 'max'
        }
        
        existing_metrics = {k: v for k, v in metrics_rules.items() if k in win_df.columns}
        
        if existing_metrics:
            tabs = st.tabs([m.replace('horizon_', '').replace('_median_median', '').replace('_', ' ').upper() for m in existing_metrics.keys()])
            
            for tab, (metric, rule) in zip(tabs, existing_metrics.items()):
                with tab:
                    clean_combo_df = win_df.dropna(subset=[metric])
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
                    ).properties(width='container', height=400, title=f"Win Rates based on {metric.replace('horizon_', '')}")
                    
                    st.altair_chart(apply_thesis_theme(chart), use_container_width=True)

        st.markdown("---")
        # --- Thesis Export: Metric Comparison ---
        st.subheader("Thesis Export: Metric Comparison")
        st.markdown("Select parameters below to generate a publication-ready bar chart.")

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        exp_size = exp_col1.selectbox("Select Size (Export)", all_sizes, key="exp_size")
        valid_exp_combos = [cfg[1] for cfg in available_configs if cfg[0] == exp_size]
        exp_combo = exp_col2.selectbox("Select Combination (Export)", valid_exp_combos, key="exp_combo")

        exp_files = grouped_files[(exp_size, exp_combo)]
        exp_df = process_configuration(exp_files)

        available_exp_metrics = {}
        if 'horizon_rmae_median' in exp_df.columns: 
            available_exp_metrics['rMAE (Median)'] = ('horizon_rmae_median', 'median')
        if 'horizon_rmse_median' in exp_df.columns: 
            available_exp_metrics['RMSE (Median)'] = ('horizon_rmse_median', 'median')
        if 'horizon_smape_median_median' in exp_df.columns: 
            available_exp_metrics['sMAPE (Median)'] = ('horizon_smape_median_median', 'median')
        elif 'horizon_smape_median' in exp_df.columns: 
            available_exp_metrics['sMAPE (Median)'] = ('horizon_smape_median', 'median')
        if 'horizon_da_mean' in exp_df.columns: 
            available_exp_metrics['DA (Mean)'] = ('horizon_da_mean', 'mean')

        exp_metric_label = exp_col3.selectbox("Select Core Metric", list(available_exp_metrics.keys()), key="exp_metric")

        if exp_metric_label:
            raw_metric, agg_func = available_exp_metrics[exp_metric_label]
            
            exp_summary = exp_df.groupby('model')[raw_metric].agg(agg_func).reset_index()
            
            sort_order = '-y' if exp_metric_label == 'DA (Mean)' else 'y'
            
            exp_color_condition = alt.condition(
                alt.FieldOneOfPredicate(field='model', oneOf=BASELINE_MODELS),
                alt.value('#ff7f0e'), 
                alt.value('#1f77b4')  
            )

            export_chart = alt.Chart(exp_summary).mark_bar().encode(
                x=alt.X('model:N', title='Model', sort=sort_order),
                y=alt.Y(f'{raw_metric}:Q', title=exp_metric_label),
                color=exp_color_condition
            ).properties(
                height=500,
                title=f"{exp_metric_label} Comparison - Size: {exp_size.title()} | Combo: {exp_combo.upper()}"
            )

            st.altair_chart(apply_thesis_theme(export_chart), use_container_width=True)
                
    else:
        st.warning("No metrics files found in the specified directory.")
else:
    st.error("Invalid directory path.")