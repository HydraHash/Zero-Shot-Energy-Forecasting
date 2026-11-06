import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import re
from pathlib import Path

BASELINE_MODELS = ['HistoricAvg', 'SeasonNaive', 'TSMixer', 'CNN']

MODEL_NAME_TO_TXT = {
    'TSMixer': '1_TSMixer',
    'CNN': '2_GoogleCNN',
    'GoogleCNN': '2_GoogleCNN',
    'SeasonNaive': '3_SeasonNaive',
    'HistoricAvg': '4_HistoricAvg',
    'Chronos2': '6_Chronos2',
    'TimeLLM': '7_TimeLLM',
    'TimesFM25': '8_TimesFM25',
    'Moirai10': '9_Moirai10',
    'Moirai11': '10_Moirai11',
    'Moirai20': '11_Moirai20',
    'TimerBase': '12_TimerBase',
    'LagLlama': '13_LagLLama',
    'LagLLama': '13_LagLLama',
    'Tirex': '14_Tirex',
    'Kairos10m': '15_Kairos10',
    'Kairos23m': '16_Kairos23',
    'kairos23m': '16_Kairos23',
    'Kairos50m': '17_Kairos50',
    'kairos50m': '17_Kairos50',
    'Sundial': '18_Sundial128',
    'Bolt9m': '19_ChronosBolt9',
    'Bolt21m': '20_ChronosBolt21',
    'Bolt43m': '21_ChronosBolt43',
    'Moment1': '22_Moment1',
    'TimesFM30': '24_TimesFM30',
}


def parse_duration_to_seconds(time_str: str) -> float | None:
    if not time_str or time_str.strip() == '-':
        return None
    clean = re.sub(r'\[.*?\]|\(!\)', '', time_str).strip()
    if not clean or clean == '-':
        return None

    h_match = re.search(r'(\d+)\s*h', clean)
    m_match = re.search(r'(\d+)\s*m', clean)
    s_match = re.search(r'(\d+)\s*s', clean)

    if not (h_match or m_match or s_match):
        return None

    hours = int(h_match.group(1)) if h_match else 0
    minutes = int(m_match.group(1)) if m_match else 0
    seconds = int(s_match.group(1)) if s_match else 0

    total = hours * 3600 + minutes * 60 + seconds
    return float(total) if total > 0 else None


def load_inference_times(metrics_dir: str = "") -> dict:
    candidate_paths = [
        Path(__file__).resolve().parent / "inference_time_overview.txt",
        Path(__file__).resolve().parent / "execution_time_overview.txt",
        Path(metrics_dir) / "execution_time_overview.txt",
        Path(metrics_dir) / "inference_time_overview.txt",
        Path(r"C:\Users\lsb-ps\Documents\zs_implementation\analysis\inference_time_overview.txt"),
        Path(r"C:\Users\lsb-ps\Documents\predictions_overview_lean\execution_time_overview.txt"),
    ]
    target_file = None
    for p in candidate_paths:
        if p and p.exists():
            target_file = p
            break

    if not target_file:
        return {}

    try:
        lines = target_file.read_text(encoding='utf-8').splitlines()
    except Exception:
        try:
            lines = target_file.read_text(encoding='latin-1').splitlines()
        except Exception:
            return {}

    inference_dict = {}
    combo_headers = []
    for line in lines:
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 7 and 'Model Name' in parts[1]:
            combo_headers = [c.lower() for c in parts[3:]]
            continue
        if combo_headers and len(parts) >= 3 + len(combo_headers):
            model_name = parts[1]
            size = parts[2].lower()
            for idx, combo in enumerate(combo_headers):
                time_str = parts[3 + idx]
                secs = parse_duration_to_seconds(time_str)
                if secs is not None:
                    inference_dict[(model_name, size, combo)] = secs
    return inference_dict


def get_inference_seconds(inference_dict: dict, model_name: str, size: str, combo: str) -> float | None:
    if not inference_dict:
        return None
    txt_name = MODEL_NAME_TO_TXT.get(model_name, model_name)
    key = (txt_name, size.lower(), combo.lower())
    if key in inference_dict:
        return inference_dict[key]
    cleaned_target = re.sub(r'^\d+_', '', model_name).lower()
    for (m_txt, s_txt, c_txt), secs in inference_dict.items():
        if s_txt == size.lower() and c_txt == combo.lower():
            cleaned_txt = re.sub(r'^\d+_', '', m_txt).lower()
            if cleaned_target == cleaned_txt or cleaned_target in cleaned_txt or cleaned_txt in cleaned_target:
                return secs
    return None

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


@st.cache_data
def load_masterdata(filepath: str):
    master_path = Path(filepath)
    if master_path.exists():
        sep = "\t" if master_path.suffix in ['.tsv', '.csv'] else ","
        df_master = pd.read_csv(master_path, sep=sep) if master_path.suffix != '.parquet' else pd.read_parquet(master_path)
        
        if 'Id' in df_master.columns:
            df_master.rename(columns={'Id': 'id'}, inplace=True)
            
        if 'Sector_group_id' in df_master.columns and 'Sector_group' in df_master.columns:
            df_master['Sector_Label'] = df_master['Sector_group_id'].astype(str) + ": " + df_master['Sector_group'].astype(str)
            
        # CV and repeatability_score are extracted if they exist
        cols_to_keep = ['id']
        if 'Sector_Label' in df_master.columns: cols_to_keep.append('Sector_Label')
        if 'CV' in df_master.columns: cols_to_keep.append('CV')
        if 'repeatability_score' in df_master.columns: cols_to_keep.append('repeatability_score')
            
        return df_master[cols_to_keep].drop_duplicates()
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
master_dir = r"C:\Users\lsb-ps\Documents\VEA_Playground\zero_shot_data\08_zero_shot_output\master_data_cv_repeat.csv"
if Path(metrics_dir).exists():
    df_metrics = load_metrics_data(metrics_dir)
    df_wilcoxon = load_wilcoxon_stats(metrics_dir)
    df_master = load_masterdata(master_dir)
    df_inference_times = load_inference_times(metrics_dir)
    
    if not df_metrics.empty:
        if not df_master.empty and 'Sector_Label' in df_master.columns:
            df_metrics = df_metrics.merge(df_master, on='id', how='left')
            df_metrics['Sector_Label'] = df_metrics['Sector_Label'].fillna('Unknown')
        else:
            df_metrics['Sector_Label'] = 'All Sectors'
            
        all_sizes = df_metrics['size_category'].unique().tolist()
        all_sectors = ["All Sectors"] + sorted(df_metrics[df_metrics['Sector_Label'] != 'Unknown']['Sector_Label'].unique().tolist())

        st.markdown("---")
        # --- Aggregated Metrics Summary ---
        st.subheader("Aggregated Metrics Summary & Comparison")

        agg_col1, agg_col2, agg_col3, agg_col4 = st.columns(4)
        size_agg = agg_col1.selectbox("Select Size (Aggregated)", all_sizes, key="size_agg")
        
        valid_combos_agg = df_metrics[df_metrics['size_category'] == size_agg]['combo_prefix'].unique().tolist()
        combo_agg = agg_col2.selectbox("Select Combination (Aggregated)", valid_combos_agg, key="combo_agg")

        sector_agg = agg_col3.selectbox("Select Sector (Aggregated)", all_sectors, key="sector_agg")

        # Dynamic Masking for Sector
        cohort_mask = (df_metrics['size_category'] == size_agg) & (df_metrics['combo_prefix'] == combo_agg)
        if sector_agg != "All Sectors":
            cohort_mask = cohort_mask & (df_metrics['Sector_Label'] == sector_agg)

        agg_df = df_metrics[cohort_mask].replace([float('inf'), float('-inf')], pd.NA)
        
        total_companies_agg = agg_df['id'].nunique()
        st.write(f"**Evaluating {total_companies_agg:,} Companies**")

        available_models_agg = agg_df['model_name'].unique().tolist()
        default_base_idx = available_models_agg.index('SeasonNaive') if 'SeasonNaive' in available_models_agg else 0
        baseline_agg = agg_col4.selectbox("Select Baseline for % Change", available_models_agg, index=default_base_idx, key="baseline_agg")

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
                    x=alt.X('Combination:N', title='Forecast Combination'),
                    y=alt.Y(f'{y_col}:Q', title=f'{y_col}')
                ).properties(width='container', height=400, title=f"{target_model} - {y_col} Across Horizons")
                return apply_thesis_theme(chart)
            
            with tab_rmae:
                st.altair_chart(create_line_chart(plot_df, "rMAE (Median)"), use_container_width=True)
            with tab_smape:
                st.altair_chart(create_line_chart(plot_df, "sMAPE (Median)"), use_container_width=True)


        st.markdown("---")
        # --- c2 vs. c3 Comparison Across Models ---
        st.subheader("C2 vs. C3 Horizon Transition Analysis")
        st.markdown(
            "Direct comparison of model performance shifts between forecast combination **c2** and **c3** "
            "on the full normal dataset across all available models."
        )

        df_c2_c3_norm = df_metrics[
            (df_metrics['size_category'] == 'normal') &
            (df_metrics['combo_prefix'].isin(['c2', 'c3']))
        ].replace([float('inf'), float('-inf')], pd.NA)

        models_c2 = set(df_c2_c3_norm[df_c2_c3_norm['combo_prefix'] == 'c2']['model_name'].unique())
        models_c3 = set(df_c2_c3_norm[df_c2_c3_norm['combo_prefix'] == 'c3']['model_name'].unique())
        common_c2_c3_models = sorted([m for m in models_c2.intersection(models_c3) if m not in BASELINE_MODELS])

        if not common_c2_c3_models:
            st.info("No models with both c2 and c3 availability found in the normal dataset.")
        else:
            smape_c2_c3_col = 'horizon_smape_median_median' if 'horizon_smape_median_median' in df_c2_c3_norm.columns else 'horizon_smape_median'
            
            df_c2_c3_filtered = df_c2_c3_norm[df_c2_c3_norm['model_name'].isin(common_c2_c3_models)]
            
            agg_c2_c3 = df_c2_c3_filtered.groupby(['model_name', 'combo_prefix']).agg({
                'horizon_rmae_median': 'median',
                smape_c2_c3_col: 'median'
            }).reset_index()

            agg_c2_c3.rename(columns={
                'model_name': 'Model',
                'combo_prefix': 'Combination',
                'horizon_rmae_median': 'rMAE (Median)',
                smape_c2_c3_col: 'sMAPE (Median)'
            }, inplace=True)

            def create_c2_c3_comparison_chart(df, metric_col, title_text, show_legend=True):
                chart = alt.Chart(df).mark_line(
                    point=alt.OverlayMarkDef(size=50)
                ).encode(
                    x=alt.X(
                        'Combination:N',
                        title='Forecast Combination',
                        sort=['c2', 'c3'],
                        axis=alt.Axis(labelAngle=0)
                    ),
                    y=alt.Y(
                        f'{metric_col}:Q',
                        title=metric_col,
                        scale=alt.Scale(zero=False)
                    ),
                    color=alt.Color(
                        'Model:N',
                        title='Model',
                        scale=alt.Scale(scheme='tableau20'),
                        legend=alt.Legend(title='Model') if show_legend else None
                    ),
                    tooltip=[
                        alt.Tooltip('Model:N', title='Model'),
                        alt.Tooltip('Combination:N', title='Combination'),
                        alt.Tooltip(f'{metric_col}:Q', title=metric_col, format='.4f')
                    ]
                ).properties(
                    width='container',
                    height=440,
                    title=alt.TitleParams(
                        text=title_text,
                        anchor='middle',
                        fontSize=18,
                        offset=15,
                        orient='top'
                    )
                )
                return apply_thesis_theme(chart)

            col_c2_c3_rmae, col_c2_c3_smape = st.columns(2)
            with col_c2_c3_rmae:
                rmae_chart = create_c2_c3_comparison_chart(
                    agg_c2_c3,
                    "rMAE (Median)",
                    "rMAE c2 to c3 Transition",
                    show_legend=False
                )
                st.altair_chart(rmae_chart, use_container_width=True)

            with col_c2_c3_smape:
                smape_chart = create_c2_c3_comparison_chart(
                    agg_c2_c3,
                    "sMAPE (Median)",
                    "sMAPE c2 to c3 Transition",
                    show_legend=True
                )
                st.altair_chart(smape_chart, use_container_width=True)

            with st.expander("View c2 vs. c3 Transition Details"):
                piv_rmae = agg_c2_c3.pivot(index='Model', columns='Combination', values='rMAE (Median)')
                piv_smape = agg_c2_c3.pivot(index='Model', columns='Combination', values='sMAPE (Median)')
                
                table_c2_c3 = pd.DataFrame({
                    'rMAE (c2)': piv_rmae['c2'],
                    'rMAE (c3)': piv_rmae['c3'],
                    'Δ rMAE (%)': ((piv_rmae['c3'] - piv_rmae['c2']) / piv_rmae['c2']) * 100,
                    'sMAPE (c2)': piv_smape['c2'],
                    'sMAPE (c3)': piv_smape['c3'],
                    'Δ sMAPE (% pts)': piv_smape['c3'] - piv_smape['c2'],
                    'Δ sMAPE (%)': ((piv_smape['c3'] - piv_smape['c2']) / piv_smape['c2']) * 100
                })
                st.dataframe(
                    table_c2_c3.style.format({
                        'rMAE (c2)': '{:.4f}',
                        'rMAE (c3)': '{:.4f}',
                        'Δ rMAE (%)': '{:+.2f}%',
                        'sMAPE (c2)': '{:.2f}%',
                        'sMAPE (c3)': '{:.2f}%',
                        'Δ sMAPE (% pts)': '{:+.2f}% pts',
                        'Δ sMAPE (%)': '{:+.2f}%',
                    }),
                    use_container_width=True
                )


        st.markdown("---")
        # --- Cross-Combination Error vs. Inference Time Analysis ---
        st.subheader("Cross-Combination Error vs. Inference Time Analysis")
        st.markdown(
            "Analyzes error metric progression across forecast combinations alongside model inference execution time."
        )

        time_col1, time_col2, time_col3, time_col4 = st.columns(4)
        size_time = time_col1.selectbox("Select Size (Time Analysis)", all_sizes, key="size_time")

        valid_combos_time = sorted(
            df_metrics[df_metrics['size_category'] == size_time]['combo_prefix'].unique().tolist(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x
        )
        selected_combos_time = time_col2.multiselect(
            "Select Contained Combinations",
            options=valid_combos_time,
            default=valid_combos_time,
            key="combos_time"
        )

        all_models_time = sorted(df_metrics[df_metrics['size_category'] == size_time]['model_name'].unique().tolist())
        target_model_time = time_col3.selectbox("Select Model (Time Analysis)", all_models_time, key="model_time")

        selected_metric_time = time_col4.selectbox(
            "Select Error Metric",
            ["rMAE (Median)", "sMAPE (Median)"],
            key="metric_time"
        )

        if not selected_combos_time:
            st.warning("Please select at least one combination.")
        else:
            combo_time_comparisons = []
            for combo in selected_combos_time:
                df_slice = df_metrics[
                    (df_metrics['size_category'] == size_time) & 
                    (df_metrics['combo_prefix'] == combo) & 
                    (df_metrics['model_name'] == target_model_time)
                ].replace([float('inf'), float('-inf')], pd.NA)

                metric_val = pd.NA
                if not df_slice.empty:
                    if selected_metric_time == "rMAE (Median)":
                        metric_val = df_slice['horizon_rmae_median'].median() if 'horizon_rmae_median' in df_slice.columns else pd.NA
                    else:  # "sMAPE (Median)"
                        if 'horizon_smape_median_median' in df_slice.columns:
                            metric_val = df_slice['horizon_smape_median_median'].median()
                        elif 'horizon_smape_median' in df_slice.columns:
                            metric_val = df_slice['horizon_smape_median'].median()

                inf_secs = get_inference_seconds(df_inference_times, target_model_time, size_time, combo)

                # Compute average inference time per company (total run time divided by dataset size)
                cohort_dataset_size = df_metrics[df_metrics['size_category'] == size_time]['id'].nunique()
                n_companies = df_slice['id'].nunique() if not df_slice.empty and df_slice['id'].nunique() > 0 else cohort_dataset_size
                avg_inf_secs = (inf_secs / n_companies) if (inf_secs is not None and n_companies > 0) else None

                combo_time_comparisons.append({
                    "Combination": combo,
                    "metric_value": metric_val,
                    "avg_time_per_company": avg_inf_secs,
                    "total_run_time": inf_secs,
                    "company_count": n_companies
                })

            if combo_time_comparisons:
                df_plot_time = pd.DataFrame(combo_time_comparisons)
                df_plot_time_valid = df_plot_time.dropna(subset=['metric_value'])

                if df_plot_time_valid.empty:
                    st.info(f"No metric data available for {target_model_time} across selected combinations.")
                else:
                    has_time_values = df_plot_time_valid['avg_time_per_company'].dropna().count() > 0

                    metric_color = '#1f77b4'  # Steel Blue
                    time_color = '#d62728'    # Crimson Red

                    base_chart = alt.Chart(df_plot_time_valid).encode(
                        x=alt.X('Combination:N', title='Forecast Combination', sort=selected_combos_time)
                    )

                    metric_line = base_chart.mark_line(
                        color=metric_color,
                        point=alt.OverlayMarkDef(color=metric_color, size=60)
                    ).encode(
                        y=alt.Y(
                            'metric_value:Q',
                            title=selected_metric_time,
                            axis=alt.Axis(
                                titleColor=metric_color,
                                labelColor=metric_color,
                                tickColor=metric_color
                            )
                        ),
                        tooltip=[
                            alt.Tooltip('Combination:N', title='Combination'),
                            alt.Tooltip('metric_value:Q', title=selected_metric_time, format='.4f')
                        ]
                    )

                    title_text_dual = f"{target_model_time} - {selected_metric_time} vs. Avg. Inference Time per Company ({size_time.title()})"
                    title_text_single = f"{target_model_time} - {selected_metric_time} Across Horizons ({size_time.title()})"

                    if has_time_values:
                        time_df = df_plot_time_valid.dropna(subset=['avg_time_per_company'])
                        time_line = alt.Chart(time_df).mark_line(
                            color=time_color,
                            point=alt.OverlayMarkDef(color=time_color, size=60)
                        ).encode(
                            x=alt.X('Combination:N', sort=selected_combos_time),
                            y=alt.Y(
                                'avg_time_per_company:Q',
                                title='Avg. Inference Time / Industrial Site (s)',
                                axis=alt.Axis(
                                    titleColor=time_color,
                                    labelColor=time_color,
                                    tickColor=time_color,
                                    orient='right',
                                    grid=False
                                )
                            ),
                            tooltip=[
                                alt.Tooltip('Combination:N', title='Combination'),
                                alt.Tooltip('avg_time_per_company:Q', title='Avg. Time / Company (s)', format='.4f'),
                                alt.Tooltip('total_run_time:Q', title='Total Run Time (s)', format='.1f'),
                                alt.Tooltip('company_count:Q', title='Dataset Size (Companies)', format=',.0f')
                            ]
                        )

                        dual_chart = alt.layer(metric_line, time_line).resolve_scale(y='independent').resolve_axis(y='independent').properties(
                            width='container',
                            height=420,
                            title=alt.TitleParams(
                                text=title_text_dual,
                                anchor='middle',
                                fontSize=20,
                                offset=15,
                                orient='top'
                            )
                        )
                        st.caption(f"🔵 **{selected_metric_time}** (Left axis) &nbsp;|&nbsp; 🔴 **Avg. Inference Time per Company (s)** (Right axis)")
                        st.altair_chart(apply_thesis_theme(dual_chart), use_container_width=True)
                    else:
                        single_chart = metric_line.properties(
                            width='container',
                            height=420,
                            title=alt.TitleParams(
                                text=title_text_single,
                                anchor='middle',
                                fontSize=20,
                                offset=15,
                                orient='top'
                            )
                        )
                        st.caption(f"🔵 **{selected_metric_time}** (Left axis)")
                        st.altair_chart(apply_thesis_theme(single_chart), use_container_width=True)
                        st.info(f"No inference time data found in 'inference_time_overview.txt' for {target_model_time} (Size: {size_time.title()}). Displaying metric line only.")

                    with st.expander("View Data Details"):
                        display_df = df_plot_time.rename(columns={
                            'metric_value': selected_metric_time,
                            'avg_time_per_company': 'Avg. Time per Company (s)',
                            'total_run_time': 'Total Run Time (s)',
                            'company_count': 'Dataset Size (Companies)'
                        })
                        st.dataframe(display_df.set_index('Combination'), use_container_width=True)


        st.markdown("---")
        # --- Win Rates Per Metric ---
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

        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)

        exp_size = exp_col1.selectbox("Select Size (Export)", all_sizes, key="exp_size")
        
        valid_exp_combos = df_metrics[df_metrics['size_category'] == exp_size]['combo_prefix'].unique().tolist()
        exp_combo = exp_col2.selectbox("Select Combination (Export)", valid_exp_combos, key="exp_combo")
        
        sector_exp = exp_col3.selectbox("Select Sector (Export)", all_sectors, key="sector_exp")

        # Dynamic Masking for Sector
        exp_mask = (df_metrics['size_category'] == exp_size) & (df_metrics['combo_prefix'] == exp_combo)
        if sector_exp != "All Sectors":
            exp_mask = exp_mask & (df_metrics['Sector_Label'] == sector_exp)

        exp_df = df_metrics[exp_mask].replace([float('inf'), float('-inf')], pd.NA)

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

        exp_metric_label = exp_col4.selectbox("Select Core Metric", list(available_exp_metrics.keys()), key="exp_metric")

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
                title=f"{exp_metric_label} - Size: {exp_size.title()} | Combo: {exp_combo.upper()} | Sector: {sector_exp}"
            )

            st.altair_chart(apply_thesis_theme(export_chart), use_container_width=True)


        st.markdown("---")
        # --- Demand Repeatability vs. Model Performance ---
        st.subheader("Demand Repeatability vs. Zero-Shot Competitiveness")
        st.markdown(
            "Evaluates whether Zero-Shot foundation models retain a comparative advantage over baseline models "
            "across different levels of intraday load repeatability (erratic vs. rigid diurnal patterns)."
        )

        rep_col1, rep_col2, rep_col3 = st.columns(3)
        size_rep = rep_col1.selectbox("Select Size (Repeatability)", all_sizes, key="size_rep")
        valid_combos_rep = df_metrics[df_metrics['size_category'] == size_rep]['combo_prefix'].unique().tolist()
        combo_rep = rep_col2.selectbox("Select Combination (Repeatability)", valid_combos_rep, key="combo_rep")
        metric_rep_choice = rep_col3.selectbox(
            "Select Evaluation Metric",
            ["rMAE (Median)", "sMAPE (Median)"],
            key="metric_rep"
        )

        if not df_master.empty and 'repeatability_score' in df_master.columns:
            rep_slice = df_metrics[
                (df_metrics['size_category'] == size_rep) &
                (df_metrics['combo_prefix'] == combo_rep)
            ].replace([float('inf'), float('-inf')], pd.NA)

            # If repeatability_score already merged into df_metrics at top-level, reuse it; otherwise merge from df_master
            if 'repeatability_score' in rep_slice.columns:
                rep_merged = rep_slice.dropna(subset=['repeatability_score'])
            else:
                rep_merged = rep_slice.merge(df_master[['id', 'repeatability_score']], on='id', how='inner')
                rep_merged = rep_merged.dropna(subset=['repeatability_score'])

            if not rep_merged.empty:
                raw_metric_col = 'horizon_rmae_median' if metric_rep_choice == "rMAE (Median)" else (
                    'horizon_smape_median_median' if 'horizon_smape_median_median' in rep_merged.columns else 'horizon_smape_median'
                )

                rep_merged = rep_merged.dropna(subset=[raw_metric_col])
                # Ensure ordered categorical tier labels
                tier_order = ['Low (<0.6)', 'Medium (0.6-0.85)', 'High (>=0.85)']
                rep_merged['tier'] = pd.cut(
                    rep_merged['repeatability_score'],
                    bins=[-np.inf, 0.60, 0.85, np.inf],
                    labels=tier_order,
                    ordered=True
                )
                rep_merged['is_baseline'] = rep_merged['model_name'].isin(BASELINE_MODELS)
                rep_merged['model_type'] = rep_merged['is_baseline'].map({True: 'Baseline', False: 'Zero-Shot'})

                # --- Plot 1: Performance Metric across Repeatability Tiers by Model ---
                tier_agg = rep_merged.groupby(['model_name', 'tier', 'model_type'], observed=False)[raw_metric_col].median().reset_index()
                tier_agg.rename(columns={raw_metric_col: 'Metric Value', 'model_name': 'Model', 'model_type': 'Model Type'}, inplace=True)
                tier_agg['tier'] = pd.Categorical(tier_agg['tier'], categories=tier_order, ordered=True)

                metric_tier_chart = alt.Chart(tier_agg).mark_circle(
                    size=90,
                    opacity=0.85
                ).encode(
                    x=alt.X(
                        'tier:O',
                        title='Intraday Repeatability Tier',
                        sort=tier_order,
                        scale=alt.Scale(domain=tier_order),
                        axis=alt.Axis(labelAngle=-30)
                    ),
                    y=alt.Y('Metric Value:Q', title=metric_rep_choice, scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'Model Type:N',
                        legend=None,
                        scale=alt.Scale(
                            domain=['Zero-Shot', 'Baseline'],
                            range=['#1f77b4', '#ff7f0e']
                        )
                    ),
                    tooltip=[
                        alt.Tooltip('Model:N', title='Model'),
                        alt.Tooltip('Model Type:N', title='Architecture'),
                        alt.Tooltip('tier:N', title='Repeatability Tier'),
                        alt.Tooltip('Metric Value:Q', title=metric_rep_choice, format='.4f')
                    ]
                ).properties(
                    width='container',
                    height=440,
                    title=alt.TitleParams(
                        text=f"{metric_rep_choice} across Intraday Repeatability Tiers (Size: {size_rep.title()} | {combo_rep.upper()})",
                        anchor='middle',
                        fontSize=18,
                        offset=15,
                        orient='top'
                    )
                )

                # --- Plot 2: Zero-Shot vs. Baseline Win Share across Tiers ---
                piv_wins = rep_merged.pivot_table(
                    index=['id', 'tier'],
                    columns='model_name',
                    values=raw_metric_col,
                    observed=False
                )
                best_model_per_company = piv_wins.idxmin(axis=1).reset_index(name='best_model')
                best_model_per_company['Model Type'] = best_model_per_company['best_model'].apply(
                    lambda x: 'Baseline' if x in BASELINE_MODELS else 'Zero-Shot'
                )

                win_shares = best_model_per_company.groupby(['tier', 'Model Type'], observed=False).size().reset_index(name='Company Wins')
                tier_totals = win_shares.groupby('tier', observed=False)['Company Wins'].transform('sum')
                win_shares['Win Share (%)'] = (win_shares['Company Wins'] / tier_totals) * 100
                win_shares['tier'] = pd.Categorical(win_shares['tier'], categories=tier_order, ordered=True)

                win_share_chart = alt.Chart(win_shares).mark_bar().encode(
                    x=alt.X(
                        'tier:O',
                        title='Intraday Repeatability Tier',
                        sort=tier_order,
                        scale=alt.Scale(domain=tier_order),
                        axis=alt.Axis(labelAngle=-30)
                    ),
                    y=alt.Y('Win Share (%):Q', title='Company Win Share (%)', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        'Model Type:N',
                        title='Model Architecture',
                        scale=alt.Scale(
                            domain=['Zero-Shot', 'Baseline'],
                            range=['#1f77b4', '#ff7f0e']
                        )
                    ),
                    tooltip=[
                        alt.Tooltip('tier:N', title='Repeatability Tier'),
                        alt.Tooltip('Model Type:N', title='Architecture'),
                        alt.Tooltip('Company Wins:Q', title='Wins (Count)'),
                        alt.Tooltip('Win Share (%):Q', title='Win Share', format='.1f')
                    ]
                ).properties(
                    width='container',
                    height=440,
                    title=alt.TitleParams(
                        text=f"{combo_rep.upper()} Zero-Shot (blue) vs. Baseline (orange) sMAPE Win Share",
                        anchor='middle',
                        fontSize=18,
                        offset=15,
                        orient='top'
                    )
                )

                col_rep1, col_rep2 = st.columns(2)
                with col_rep1:
                    st.altair_chart(apply_thesis_theme(metric_tier_chart), use_container_width=True)
                    st.caption("🔵 **Zero-Shot models** &nbsp;|&nbsp; 🟠 **Baseline models**.")
                with col_rep2:
                    st.altair_chart(apply_thesis_theme(win_share_chart), use_container_width=True)
                    st.caption("🔵 **Zero-Shot Models** &nbsp;|&nbsp; 🟠 **Baselines** (Company win determined by lowest error).")

                # --- Plot 3: Scatter Correlation for Selected Model ---
                with st.expander("Explore Continuous Correlation (Repeatability Score vs. Model Error)"):
                    all_avail_models = sorted(rep_merged['model_name'].unique().tolist())
                    selected_scatter_model = st.selectbox(
                        "Select Model for Detailed Scatter Plot",
                        all_avail_models,
                        key="model_scatter_rep"
                    )

                    scatter_df = rep_merged[rep_merged['model_name'] == selected_scatter_model].dropna(
                        subset=['repeatability_score', raw_metric_col]
                    )

                    if not scatter_df.empty:
                        r_score = scatter_df['repeatability_score'].corr(scatter_df[raw_metric_col])
                        st.info(f"**Pearson Correlation ($r$) for {selected_scatter_model}:** `{r_score:.4f}`")

                        scatter_plot = alt.Chart(scatter_df).mark_circle(size=40, opacity=0.4, color='#1f77b4').encode(
                            x=alt.X('repeatability_score:Q', title='Daily Repeatability Score (Pearson r across days)'),
                            y=alt.Y(f'{raw_metric_col}:Q', title=metric_rep_choice),
                            tooltip=['id', 'repeatability_score', alt.Tooltip(f'{raw_metric_col}:Q', format='.4f')]
                        )
                        trendline_rep = scatter_plot.transform_regression(
                            'repeatability_score', raw_metric_col
                        ).mark_line(color='#ff7f0e', strokeWidth=3)

                        scatter_chart = (scatter_plot + trendline_rep).properties(
                            width='container',
                            height=450,
                            title=alt.TitleParams(
                                text=f"{selected_scatter_model}: {metric_rep_choice} vs. Daily Repeatability Score",
                                anchor='middle',
                                fontSize=18,
                                offset=15,
                                orient='top'
                            )
                        ).interactive()

                        st.altair_chart(apply_thesis_theme(scatter_chart), use_container_width=True)
            else:
                st.warning("No matching metrics found for the selected combination and size.")
        else:
            st.error("The 'repeatability_score' column was not found in the master data.")


        st.markdown("---")
        # --- Predictability Distribution ---
        st.subheader("Predictability Distribution: Variance vs. Error Floor")
        st.markdown("Analyzes the correlation between inherent time-series volatility (CV) and the empirical lower bound of predictability (lowest achieved error).")

        pred_col1, pred_col2 = st.columns(2)
        size_pred = pred_col1.selectbox("Select Size (Predictability)", all_sizes, key="size_pred")
        
        valid_combos_pred = df_metrics[df_metrics['size_category'] == size_pred]['combo_prefix'].unique().tolist()
        combo_pred = pred_col2.selectbox("Select Combination (Predictability)", valid_combos_pred, key="combo_pred")

        pred_df = df_metrics[
            (df_metrics['size_category'] == size_pred) & 
            (df_metrics['combo_prefix'] == combo_pred)
        ].copy()

        # Identify the core sMAPE column
        smape_col = 'horizon_smape_median_median' if 'horizon_smape_median_median' in pred_df.columns else 'horizon_smape_median'
        
        # Calculate Empirical Lower Bound per ID
        lower_bound_df = pred_df.groupby('id')[smape_col].min().reset_index()
        lower_bound_df.rename(columns={smape_col: 'Minimum_sMAPE'}, inplace=True)

        if not df_master.empty and 'CV' in df_master.columns:
            # Merge with Masterdata to pair CV with the minimum achieved error
            merged_pred = lower_bound_df.merge(df_master[['id', 'CV']], on='id', how='inner')
            merged_pred = merged_pred[merged_pred['CV'] >= 0].dropna(subset=['CV', 'Minimum_sMAPE'])
            
            if not merged_pred.empty:
                # Calculate Pearson Correlation
                pearson_corr = merged_pred['CV'].corr(merged_pred['Minimum_sMAPE'])
                
                st.info(f"**Pearson Correlation Coefficient ($r$):** `{pearson_corr:.4f}`")
                
                # Scatter Plot with Regression Trendline
                scatter = alt.Chart(merged_pred).mark_circle(size=40, opacity=0.4, color='#1f77b4').encode(
                    x=alt.X('CV:Q', title='Coefficient of Variation (CV)'),
                    y=alt.Y('Minimum_sMAPE:Q', title='Empirical Lower Bound (Min sMAPE)'),
                    tooltip=['id', 'CV', 'Minimum_sMAPE']
                )
                
                trendline = scatter.transform_regression('CV', 'Minimum_sMAPE').mark_line(color='#ff7f0e', strokeWidth=3)
                
                pred_chart = (scatter + trendline).properties(
                    height=500,
                    title=f"Predictability Bound vs. Variance (Size: {size_pred.title()} | Combo: {combo_pred.upper()})"
                ).interactive()
                
                st.altair_chart(apply_thesis_theme(pred_chart), use_container_width=True)
            else:
                st.warning("No valid overlapping IDs found between the metrics and the masterdata CV column.")
        else:
            st.error("The Coefficient of Variation column was not found in the masterdata.")
                
    else:
        st.warning("Master metrics parquet file is empty or not generated yet.")
else:
    st.error("Invalid directory path. Please provide the root directory containing 'master_metrics.parquet'.")