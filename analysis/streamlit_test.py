import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

BASELINE_MODELS = ['HistoricAvg', 'SeasonNaive', 'TSMixer', 'CNN']

@st.cache_data
def load_metrics_data(base_dir: str):
    base_path = Path(base_dir)
    metrics_file = base_path / "master_metrics.parquet"
    if metrics_file.exists():
        return pd.read_parquet(metrics_file, engine='pyarrow')
    return pd.DataFrame()

@st.cache_data
def load_wilcoxon_stats(base_dir: str):
    base_path = Path(base_dir)
    wilcoxon_file = base_path / "master_wilcoxon_tests.parquet"
    if wilcoxon_file.exists():
        return pd.read_parquet(wilcoxon_file, engine='pyarrow')
    return pd.DataFrame()

def apply_thesis_theme(chart):
    return chart.configure(
        background='white'
    ).configure_axis(
        labelColor='black',
        titleColor='black',
        gridColor='#e5e5e5',
        domainColor='black',
        tickColor='black',
        labelFontSize=13,
        titleFontSize=18
    ).configure_title(
        color='black',
        fontSize=20,
        anchor='middle'
    ).configure_legend(
        titleColor='black',
        labelColor='black'
    )


st.set_page_config(page_title="ZS Model Dashboard", layout="wide")
st.title("Energy Load Forecast Evaluation")
metrics_dir = r"C:\Users\lsb-ps\Documents\predictions_overview_lean"

if Path(metrics_dir).exists():
    df_metrics = load_metrics_data(metrics_dir)
    df_wilcoxon = load_wilcoxon_stats(metrics_dir)
    
    if not df_metrics.empty:
        # Extract available configurations directly from the master dataframe
        all_sizes = df_metrics['size_category'].unique().tolist()
        
        st.markdown("---")
        # --- Aggregated Metrics Summary ---
        st.subheader("Aggregated Metrics Summary & Comparison")

        agg_col1, agg_col2, agg_col3 = st.columns(3)
        size_agg = agg_col1.selectbox("Select Size (Aggregated)", all_sizes, key="size_agg")
        
        valid_combos_agg = df_metrics[df_metrics['size_category'] == size_agg]['combo_prefix'].unique().tolist()
        combo_agg = agg_col2.selectbox("Select Combination (Aggregated)", valid_combos_agg, key="combo_agg")

        # Isolate the cohort and replace infinite values with NA for safe aggregation
        agg_df = df_metrics[
            (df_metrics['size_category'] == size_agg) & 
            (df_metrics['combo_prefix'] == combo_agg)
        ].replace([float('inf'), float('-inf')], pd.NA)
        
        total_companies_agg = agg_df['id'].nunique()
        st.write(f"**Evaluating {total_companies_agg:,} Companies**")

        available_models_agg = agg_df['model_name'].unique().tolist()
        default_base_idx = available_models_agg.index('SeasonNaive') if 'SeasonNaive' in available_models_agg else 0
        baseline_agg = agg_col3.selectbox("Select Baseline for % Change", available_models_agg, index=default_base_idx, key="baseline_agg")

        agg_functions = {
            'horizon_rmae_median': 'median',
            'horizon_rmse_median': 'median',
            'horizon_smape_median_median': 'median', 
            'horizon_da_mean': 'mean'
        }

        valid_agg = {k: v for k, v in agg_functions.items() if k in agg_df.columns}

        summary_stats = agg_df.groupby('model_name').agg(valid_agg).reset_index()
        summary_stats.rename(columns={
            'horizon_rmae_median': 'rMAE (Median)', 
            'horizon_rmse_median': 'RMSE (Median)',
            'horizon_smape_median_median': 'sMAPE (Median)', 
            'horizon_da_mean': 'DA (Mean)'
        }, inplace=True)

        base_metrics = [col for col in summary_stats.columns if col != 'model_name']
        baseline_row = summary_stats[summary_stats['model_name'] == baseline_agg]

        display_stats = summary_stats[['model_name']].copy()

        for col in base_metrics:
            display_stats[col] = summary_stats[col] 
            
            if not baseline_row.empty:
                base_val = baseline_row[col].values[0]
                if base_val != 0:
                    display_stats[f"{col} (% Δ)"] = ((summary_stats[col] - base_val) / base_val) * 100
                else:
                    display_stats[f"{col} (% Δ)"] = pd.Series([
                        float('inf') if v > 0 else (0.0 if v == 0 else float('-inf')) 
                        for v in summary_stats[col]
                    ])

        format_dict = {col: "{:.4f}" for col in base_metrics}
        format_dict.update({f"{col} (% Δ)": "{:+.2f}%" for col in base_metrics})

        st.dataframe(
            display_stats.set_index('model_name').style.format(format_dict, na_rep="-"), 
            use_container_width=True
        )

        metric_cols = base_metrics 
        val_tabs = st.tabs(metric_cols)

        for tab, metric_col in zip(val_tabs, metric_cols):
            with tab:
                plot_df = summary_stats[['model_name', metric_col]]
                
                color_condition = alt.condition(
                    alt.FieldOneOfPredicate(field='model_name', oneOf=BASELINE_MODELS),
                    alt.value('#ff7f0e'), 
                    alt.value('#1f77b4')  
                )
                
                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X('model_name:N', title='Model Name', sort='y' if metric_col != 'DA (Mean)' else '-y'),
                    y=alt.Y(f'{metric_col}:Q', title=f'{metric_col}'),
                    color=color_condition
                ).properties(width='container', height=400, title=f"Aggregated {metric_col}")
                
                st.altair_chart(apply_thesis_theme(chart), use_container_width=True)


        st.markdown("---")
        # --- Statistical Significance Section ---
        st.subheader("Statistical Significance (Wilcoxon Signed-Rank Test)")
        
        stat_col1, stat_col2 = st.columns(2)
        size_stat = stat_col1.selectbox("Select Size (Stats)", all_sizes, key="size_stat")
        
        valid_combos_stat = df_metrics[df_metrics['size_category'] == size_stat]['combo_prefix'].unique().tolist()
        combo_stat = stat_col2.selectbox("Select Combination (Stats)", valid_combos_stat, key="combo_stat")
        
        if not df_wilcoxon.empty:
            cohort_wilcoxon = df_wilcoxon[
                (df_wilcoxon['size_category'] == size_stat) & 
                (df_wilcoxon['combo_prefix'] == combo_stat)
            ].copy()
            
            if not cohort_wilcoxon.empty:
                # Drop columns that are used purely for internal slicing
                drop_cols = ['size_category', 'combo_prefix']
                if 'stratification_type' in cohort_wilcoxon.columns:
                    drop_cols.extend(['stratification_type', 'stratification_value'])
                
                cohort_wilcoxon.drop(columns=[c for c in drop_cols if c in cohort_wilcoxon.columns], inplace=True, errors='ignore')
                
                metric_name_map = {
                    'horizon_rmae_median': 'rMAE (Median)',
                    'horizon_smape_median_median': 'sMAPE (Median)',
                    'horizon_smape_median': 'sMAPE (Median)', 
                    'horizon_rmse_median': 'RMSE (Median)',
                    'horizon_da_mean': 'DA (Mean)'
                }
                cohort_wilcoxon['target_metric'] = cohort_wilcoxon['target_metric'].map(metric_name_map).fillna(cohort_wilcoxon['target_metric'])
                
                # Dynamic column renaming based on available data from the Wilcoxon master
                rename_map = {
                    'target_metric': 'Metric',
                    'model_name': 'Model',
                    'baseline_model': 'Baseline',
                    'n_valid_samples': 'Valid (N)',
                    'dropped_infs_nans': 'Dropped (Inf/NaN)',
                    'median_difference': 'Median Diff.',
                    'win_rate_pct': 'Win Rate (%)',
                    'rank_biserial_r': 'Effect Size (r)',
                    'w_statistic': 'W-Stat',
                    'p_value': 'p-value',
                    'p_value_fdr': 'FDR p-value',
                    'is_significant': 'Significant',
                    'conclusion': 'Conclusion'
                }
                cohort_wilcoxon.rename(columns=rename_map, inplace=True)
                
                def color_significance(val):
                    if isinstance(val, bool):
                        return 'background-color: #2e7d32; color: white' if val else 'background-color: #c62828; color: white'
                    return ''
                    
                styled_df = cohort_wilcoxon.style.map(color_significance, subset=['Significant'])
                
                format_dict = {
                    'p-value': lambda x: '< 0.001' if (x < 0.001 and x >= 0) else f"{x:.4f}",
                    'FDR p-value': lambda x: '< 0.001' if (x < 0.001 and x >= 0) else f"{x:.4f}",
                    'Median Diff.': "{:.4f}",
                    'Win Rate (%)': "{:.2f}%",
                    'Effect Size (r)': "{:.3f}",
                    'W-Stat': "{:.0f}",
                    'neg_log10_p_fdr': "{:.2f}"
                }
                
                # Apply format only to columns that exist
                styled_df = styled_df.format({k: v for k, v in format_dict.items() if k in cohort_wilcoxon.columns})
                    
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No statistical tests found for Size: {size_stat}, Combo: {combo_stat}.")
        else:
            st.warning("Wilcoxon parquet file not found. Ensure 'master_wilcoxon_tests.parquet' exists in the metrics directory.")


        st.markdown("---")
        # --- Cross-Combination Analysis ---
        st.subheader("Cross-Combination Model Analysis")
        
        cross_col1, cross_col2 = st.columns(2)
        size_cross = cross_col1.selectbox("Select Size (Cross-Combo)", all_sizes, key="size_cross")
        
        valid_combos_cross = df_metrics[df_metrics['size_category'] == size_cross]['combo_prefix'].unique().tolist()
        all_models = df_metrics[df_metrics['size_category'] == size_cross]['model_name'].unique().tolist()
                
        target_model = cross_col2.selectbox("Select Model to Analyze Across Combinations", all_models, key="model_cross")
        
        combo_comparisons = []
        for combo in valid_combos_cross:
            # Isolate the exact model, size, and combo
            df_slice = df_metrics[
                (df_metrics['size_category'] == size_cross) & 
                (df_metrics['combo_prefix'] == combo) & 
                (df_metrics['model_name'] == target_model)
            ].replace([float('inf'), float('-inf')], pd.NA)
            
            if not df_slice.empty:
                rmae_val = df_slice['horizon_rmae_median'].median() if 'horizon_rmae_median' in df_slice.columns else pd.NA
                
                if 'horizon_smape_median_median' in df_slice.columns:
                    smape_val = df_slice['horizon_smape_median_median'].median()
                elif 'horizon_smape_median' in df_slice.columns:
                    smape_val = df_slice['horizon_smape_median'].median()
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
        
        valid_combos_win = df_metrics[df_metrics['size_category'] == size_win]['combo_prefix'].unique().tolist()
        combo_win = win_col2.selectbox("Select Combination (Win Rates)", valid_combos_win, key="combo_win")
        
        win_df = df_metrics[
            (df_metrics['size_category'] == size_win) & 
            (df_metrics['combo_prefix'] == combo_win)
        ]
        
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
                    clean_combo_df = win_df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=[metric])
                    pivot_df = clean_combo_df.pivot_table(index='id', columns='model_name', values=metric, aggfunc='median')            
                    
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
        # --- Metric Comparison ---
        st.subheader("Metric Comparison")
        st.markdown("Select parameters below to generate a bar chart.")

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        exp_size = exp_col1.selectbox("Select Size (Export)", all_sizes, key="exp_size")
        
        valid_exp_combos = df_metrics[df_metrics['size_category'] == exp_size]['combo_prefix'].unique().tolist()
        exp_combo = exp_col2.selectbox("Select Combination (Export)", valid_exp_combos, key="exp_combo")

        exp_df = df_metrics[
            (df_metrics['size_category'] == exp_size) & 
            (df_metrics['combo_prefix'] == exp_combo)
        ].replace([float('inf'), float('-inf')], pd.NA)

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
            exp_summary = exp_df.groupby('model_name')[raw_metric].agg(agg_func).reset_index()
            sort_order = '-y' if exp_metric_label == 'DA (Mean)' else 'y'
            
            exp_color_condition = alt.condition(
                alt.FieldOneOfPredicate(field='model_name', oneOf=BASELINE_MODELS),
                alt.value('#ff7f0e'), 
                alt.value('#1f77b4')  
            )

            bars = alt.Chart(exp_summary).mark_bar().encode(
                x=alt.X('model_name:N', title='Model', sort=sort_order, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y(f'{raw_metric}:Q', title=exp_metric_label),
                color=exp_color_condition
            )
            
            text = bars.mark_text(
                align='center',
                fontSize=11,
                baseline='bottom',
                dy=-5,
                color='black'
            ).encode(
                text=alt.Text(f'{raw_metric}:Q', format='.3f') 
            )

            export_chart = (bars + text).properties(
                height=500,
                title=f"{exp_metric_label} - Dataset Size: {exp_size.title()} | Forecast Combination: {exp_combo.upper()}"
            )

            st.altair_chart(apply_thesis_theme(export_chart), use_container_width=True)
                
    else:
        st.warning("Master metrics parquet file is empty or not generated yet.")
else:
    st.error("Invalid directory path. Please provide the root directory containing 'master_metrics.parquet'.")