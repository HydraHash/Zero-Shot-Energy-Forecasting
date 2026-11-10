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
    'Tirex': '14_Tirex',
    'Kairos10m': '15_Kairos10',
    'Kairos23m': '16_Kairos23',
    'Kairos50m': '17_Kairos50',
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
        Path(r"path/to/inference_time_overview.txt"),
        Path(r"path/to/execution_time_overview.txt"),
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
def load_seasonal_metrics(base_dir: str):
    candidate_paths = [
        Path(base_dir) / "seasonal_monthly_metrics.parquet",
        Path(__file__).resolve().parent / "seasonal_monthly_metrics.parquet",
        Path(r"path/to/seasonal_monthly_metrics.parquet"),
    ]
    for p in candidate_paths:
        if p.exists():
            return pd.read_parquet(p, engine='pyarrow')
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
            
        cols_to_keep = ['id']
        if 'Sector_group' in df_master.columns: cols_to_keep.append('Sector_group')
        if 'Sector_Label' in df_master.columns: cols_to_keep.append('Sector_Label')
        if 'CV' in df_master.columns: cols_to_keep.append('CV')
        if 'repeatability_score' in df_master.columns: cols_to_keep.append('repeatability_score')
            
        return df_master[cols_to_keep].drop_duplicates()
    return pd.DataFrame()


@st.cache_data
def load_loadprofile_features(filepath: str):
    p = Path(filepath)
    if p.exists():
        df_feat = pd.read_csv(p)
        return df_feat
    return pd.DataFrame()


@st.cache_data
def load_portfolio_metrics(metrics_dir: str):
    candidate_paths = [
        Path(metrics_dir) / "portfolio_aggregation_metrics.parquet",
        Path(__file__).resolve().parent / "portfolio_aggregation_metrics.parquet",
        Path(r"path/to/portfolio_aggregation_metrics.parquet"),
    ]
    for p in candidate_paths:
        if p.exists():
            return pd.read_parquet(p)
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
metrics_dir = r"path/to/predictions_overview_lean"
master_dir = r"path/to/master_data_cv_repeat.csv"
features_dir = r"path/to/loadprofile_features.csv"
if Path(metrics_dir).exists():
    df_metrics = load_metrics_data(metrics_dir)
    df_master = load_masterdata(master_dir)
    df_features = load_loadprofile_features(features_dir)
    df_inference_times = load_inference_times(metrics_dir)
    df_seasonal = load_seasonal_metrics(metrics_dir)
    df_portfolio = load_portfolio_metrics(metrics_dir)
    
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
        # --- C2 vs. C3 Horizon Transition Analysis ---
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
                    else:
                        if 'horizon_smape_median_median' in df_slice.columns:
                            metric_val = df_slice['horizon_smape_median_median'].median()
                        elif 'horizon_smape_median' in df_slice.columns:
                            metric_val = df_slice['horizon_smape_median'].median()

                inf_secs = get_inference_seconds(df_inference_times, target_model_time, size_time, combo)

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

                    metric_color = '#1f77b4'
                    time_color = '#d62728'

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
        # --- Company-Level Win Rates ---
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
        # --- Demand Repeatability vs. Zero-Shot Competitiveness ---
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
                tier_order = ['Low (<0.6)', 'Medium (0.6-0.85)', 'High (>=0.85)']
                rep_merged['tier'] = pd.cut(
                    rep_merged['repeatability_score'],
                    bins=[-np.inf, 0.60, 0.85, np.inf],
                    labels=tier_order,
                    ordered=True
                )
                rep_merged['is_baseline'] = rep_merged['model_name'].isin(BASELINE_MODELS)
                rep_merged['model_type'] = rep_merged['is_baseline'].map({True: 'Baseline', False: 'Zero-Shot'})

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
            else:
                st.warning("No matching metrics found for the selected combination and size.")
        else:
            st.error("The 'repeatability_score' column was not found in the master data.")


        st.markdown("---")
        # --- Peak Penalty & Outlier Volatility: rMAE vs. rRMSE ---
        st.subheader("Peak Penalty & Outlier Volatility: rMAE vs. rRMSE")
        st.markdown(
            "Compares typical day-to-day error (**rMAE**) against peak and outlier error sensitivity (**rRMSE**, relative to Seasonal Naive). "
            "Models above the diagonal line ($y = x$) suffer from a disproportionate **Peak Penalty** (catastrophic spike misses despite good average tracking), "
            "while models in the bottom-left corner excel at both general load forecasting and peak resilience."
        )

        peak_col1, peak_col2 = st.columns(2)
        size_peak = peak_col1.selectbox("Select Size (Peak Penalty)", all_sizes, key="size_peak")
        valid_combos_peak = df_metrics[df_metrics['size_category'] == size_peak]['combo_prefix'].unique().tolist()
        combo_peak = peak_col2.selectbox("Select Combination (Peak Penalty)", valid_combos_peak, key="combo_peak")

        df_peak = df_metrics[
            (df_metrics['size_category'] == size_peak) &
            (df_metrics['combo_prefix'] == combo_peak)
        ].replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['horizon_rmse_median', 'horizon_rmae_median'])

        if df_peak.empty or 'SeasonNaive' not in df_peak['model_name'].values:
            st.warning("Insufficient metric data or SeasonNaive baseline missing for this combination.")
        else:
            sn_peak = df_peak[df_peak['model_name'] == 'SeasonNaive'][['id', 'horizon_rmse_median']].rename(
                columns={'horizon_rmse_median': 'sn_rmse'}
            )
            df_peak_merged = df_peak.merge(sn_peak, on='id', how='inner')
            df_peak_merged['rRMSE'] = df_peak_merged['horizon_rmse_median'] / df_peak_merged['sn_rmse']
            df_peak_merged['rMAE'] = df_peak_merged['horizon_rmae_median']

            peak_agg = df_peak_merged.groupby('model_name').agg({
                'rMAE': 'median',
                'rRMSE': 'median'
            }).reset_index()

            peak_agg.rename(columns={'model_name': 'Model'}, inplace=True)
            peak_agg['Architecture'] = peak_agg['Model'].apply(lambda x: 'Baseline' if x in BASELINE_MODELS else 'Zero-Shot')
            peak_agg['Peak Penalty Ratio (rRMSE / rMAE)'] = peak_agg['rRMSE'] / peak_agg['rMAE']

            min_x = 0.7
            max_x = 1.8
            min_y = 0.7
            max_y = 1.9
            diag_min = max(min_x, min_y)
            diag_max = min(max_x, max_y)

            peak_agg['is_outlier'] = (peak_agg['rMAE'] > max_x) | (peak_agg['rRMSE'] > max_y) | (peak_agg['rMAE'] < min_x) | (peak_agg['rRMSE'] < min_y)
            peak_agg['display_label'] = peak_agg['Model'] + peak_agg['is_outlier'].map({True: ' [clipped]', False: ''})
            peak_agg['plot_rMAE'] = peak_agg['rMAE'].clip(lower=min_x, upper=max_x)
            peak_agg['plot_rRMSE'] = peak_agg['rRMSE'].clip(lower=min_y, upper=max_y)

            diag_line_df = pd.DataFrame({'plot_rMAE': [diag_min, diag_max], 'plot_rRMSE': [diag_min, diag_max]})
            diag_line = alt.Chart(diag_line_df).mark_line(
                color='#888888',
                strokeDash=[5, 5],
                strokeWidth=1.5
            ).encode(
                x=alt.X('plot_rMAE:Q', scale=alt.Scale(domain=[min_x, max_x])),
                y=alt.Y('plot_rRMSE:Q', scale=alt.Scale(domain=[min_y, max_y]))
            )

            diag_label = alt.Chart(pd.DataFrame({'plot_rMAE': [diag_min + (diag_max - diag_min) * 0.70], 'plot_rRMSE': [diag_min + (diag_max - diag_min) * 0.74]})).mark_text(
                text='Equilibrium (rRMSE = rMAE)',
                color='#777777',
                angle=45,
                fontSize=11,
                align='left'
            ).encode(x='plot_rMAE:Q', y='plot_rRMSE:Q')

            ref_x_df = pd.DataFrame({'plot_rMAE': [1.0]})
            ref_x_rule = alt.Chart(ref_x_df).mark_rule(
                color='#1f2937',
                strokeWidth=2.2,
                strokeDash=[5, 4]
            ).encode(x='plot_rMAE:Q')

            ref_y_df = pd.DataFrame({'plot_rRMSE': [1.0]})
            ref_y_rule = alt.Chart(ref_y_df).mark_rule(
                color='#1f2937',
                strokeWidth=2.2,
                strokeDash=[5, 4]
            ).encode(y='plot_rRMSE:Q')

            highlight_peak = alt.selection_point(fields=['Architecture'], bind='legend')

            peak_scatter = alt.Chart(peak_agg).mark_circle(size=150, opacity=0.9).encode(
                x=alt.X(
                    'plot_rMAE:Q',
                    title='Typical Day-to-Day Error: rMAE (Median)',
                    scale=alt.Scale(domain=[min_x, max_x])
                ),
                y=alt.Y(
                    'plot_rRMSE:Q',
                    title='Peak & Outlier Error: rRMSE (Median)',
                    scale=alt.Scale(domain=[min_y, max_y])
                ),
                color=alt.Color(
                    'Architecture:N',
                    title='Model Architecture',
                    scale=alt.Scale(
                        domain=['Zero-Shot', 'Baseline'],
                        range=['#1f77b4', '#ff7f0e']
                    )
                ),
                opacity=alt.condition(highlight_peak, alt.value(0.95), alt.value(0.15)),
                tooltip=[
                    alt.Tooltip('Model:N', title='Model'),
                    alt.Tooltip('Architecture:N', title='Architecture'),
                    alt.Tooltip('rMAE:Q', title='Actual rMAE (Median)', format='.4f'),
                    alt.Tooltip('rRMSE:Q', title='Actual rRMSE (Median)', format='.4f'),
                    alt.Tooltip('Peak Penalty Ratio (rRMSE / rMAE):Q', title='Peak Penalty Ratio', format='.3f')
                ]
            ).add_params(highlight_peak)

            label_pos_map = {
                'Bolt9m': 'bottom-right',
                'Bolt43m': 'bottom-right',
                'Bolt21m': 'bottom-right',
                'Chronos2': 'bottom-left',
                'Moirai20': 'top-left',
                'TSMixer': 'bottom-right',
                'CNN': 'bottom-right',
                'SeasonNaive': 'top-left'
            }
            peak_agg['label_pos'] = peak_agg['Model'].map(label_pos_map).fillna('right')

            pos_styles = {
                'top-left': {'align': 'right', 'baseline': 'bottom', 'dx': -9, 'dy': -6},
                'top-right': {'align': 'left', 'baseline': 'bottom', 'dx': 9, 'dy': -8},
                'bottom-left': {'align': 'right', 'baseline': 'top', 'dx': -9, 'dy': 6},
                'bottom-right': {'align': 'left', 'baseline': 'top', 'dx': 9, 'dy': 8},
                'right': {'align': 'left', 'baseline': 'middle', 'dx': 11, 'dy': 0}
            }

            label_layers = []
            for pos_name, style in pos_styles.items():
                pos_df = peak_agg[peak_agg['label_pos'] == pos_name]
                if not pos_df.empty:
                    halo = alt.Chart(pos_df).mark_text(
                        align=style['align'],
                        baseline=style['baseline'],
                        dx=style['dx'],
                        dy=style['dy'],
                        fontSize=11,
                        stroke='white',
                        strokeWidth=3,
                        strokeJoin='round'
                    ).encode(
                        x='plot_rMAE:Q',
                        y='plot_rRMSE:Q',
                        text='display_label:N',
                        opacity=alt.condition(highlight_peak, alt.value(1.0), alt.value(0.15))
                    )
                    text_layer = alt.Chart(pos_df).mark_text(
                        align=style['align'],
                        baseline=style['baseline'],
                        dx=style['dx'],
                        dy=style['dy'],
                        fontSize=11,
                        color='black'
                    ).encode(
                        x='plot_rMAE:Q',
                        y='plot_rRMSE:Q',
                        text='display_label:N',
                        opacity=alt.condition(highlight_peak, alt.value(1.0), alt.value(0.15))
                    )
                    label_layers.extend([halo, text_layer])

            quadrant_chart = alt.layer(
                diag_line,
                diag_label,
                ref_x_rule,
                ref_y_rule,
                peak_scatter,
                *label_layers
            ).properties(
                width='container',
                height=580,
                title=alt.TitleParams(
                    text=f"Model Peak Penalty: rMAE vs. rRMSE ({combo_peak.upper()})",
                    anchor='middle',
                    fontSize=18,
                    offset=15,
                    orient='top'
                )
            ).interactive()

            st.altair_chart(apply_thesis_theme(quadrant_chart), use_container_width=True)
            st.caption(
                "**Bottom-Left Quadrant (< 1.0, < 1.0):** Superior to SeasonNaive on both typical days and peak loads. &nbsp;|&nbsp; "
                "**Above Diagonal ($y > x$):** Higher peak penalty. &nbsp;|&nbsp; "
                "**Below Diagonal ($y < x$):** More resilient during severe spikes."
            )

            with st.expander("View Peak Penalty Metrics Table"):
                display_peak_df = peak_agg.sort_values('Peak Penalty Ratio (rRMSE / rMAE)', ascending=False).set_index('Model')
                st.dataframe(
                    display_peak_df.style.format({
                        'rMAE': '{:.4f}',
                        'rRMSE': '{:.4f}',
                        'Peak Penalty Ratio (rRMSE / rMAE)': '{:.3f}'
                    }),
                    use_container_width=True
                )


        st.markdown("---")
        # --- Trend Identification: Magnitude vs. Momentum Trade-Off ---
        st.subheader("Trend Identification: Magnitude vs. Momentum Trade-Off")
        st.markdown(
            "Analyzes whether models that achieve low magnitude error (**rMAE**, average load tracking) also capture "
            "step-to-step ramping momentum (**Directional Accuracy / DA**). "
            "The red horizontal dashed line at **50%** marks the random coin-flip threshold; the top-left green quadrant "
            "highlights models achieving both superior amplitude precision ($< 1.0$) and positive directional foresight ($> 50%$). "
            "Notice how token-quantized models (e.g. Chronos) trade off directional fidelity for level accuracy."
        )

        da_col1, da_col2 = st.columns(2)
        size_da = da_col1.selectbox("Select Size (Trend Identification)", all_sizes, key="size_da")
        valid_combos_da = df_metrics[df_metrics['size_category'] == size_da]['combo_prefix'].unique().tolist()
        combo_da = da_col2.selectbox("Select Combination (Trend Identification)", valid_combos_da, key="combo_da")

        df_da_slice = df_metrics[
            (df_metrics['size_category'] == size_da) &
            (df_metrics['combo_prefix'] == combo_da)
        ].replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['horizon_rmae_median', 'horizon_da_mean'])

        if df_da_slice.empty:
            st.warning("No Directional Accuracy or rMAE data available for this selection.")
        else:
            da_agg = df_da_slice.groupby('model_name').agg({
                'horizon_rmae_median': 'median',
                'horizon_da_mean': 'median'
            }).reset_index()

            da_agg.rename(columns={
                'model_name': 'Model',
                'horizon_rmae_median': 'rMAE',
                'horizon_da_mean': 'DA'
            }, inplace=True)

            da_agg['Architecture'] = da_agg['Model'].apply(lambda x: 'Baseline' if x in BASELINE_MODELS else 'Zero-Shot')

            min_da_x = 0.6
            max_da_x = 2.0
            min_da_y = 30.0
            max_da_y = 56.0

            da_agg['is_outlier'] = (da_agg['rMAE'] > max_da_x) | (da_agg['rMAE'] < min_da_x) | (da_agg['DA'] > max_da_y) | (da_agg['DA'] < min_da_y)
            da_agg['display_label'] = da_agg['Model'] + da_agg['is_outlier'].map({True: ' [clipped]', False: ''})
            da_agg['plot_rMAE'] = da_agg['rMAE'].clip(lower=min_da_x, upper=max_da_x)
            da_agg['plot_DA'] = da_agg['DA'].clip(lower=min_da_y, upper=max_da_y)

            green_rect_da_df = pd.DataFrame([{
                'plot_rMAE': min_da_x,
                'plot_rMAE2': 1.0,
                'plot_DA': 50.0,
                'plot_DA2': max_da_y
            }])
            green_rect_da = alt.Chart(green_rect_da_df).mark_rect(
                color='#ffffff',
                opacity=0.45
            ).encode(
                x=alt.X('plot_rMAE:Q', scale=alt.Scale(domain=[min_da_x, max_da_x])),
                x2='plot_rMAE2:Q',
                y=alt.Y('plot_DA:Q', scale=alt.Scale(domain=[min_da_y, max_da_y])),
                y2='plot_DA2:Q'
            )

            ref_da_x = alt.Chart(pd.DataFrame({'plot_rMAE': [1.0]})).mark_rule(
                color='#1f2937',
                strokeWidth=2.2,
                strokeDash=[5, 4]
            ).encode(x='plot_rMAE:Q')

            ref_da_y = alt.Chart(pd.DataFrame({'plot_DA': [50.0]})).mark_rule(
                color='#dc2626',
                strokeWidth=2.0,
                strokeDash=[4, 3]
            ).encode(y='plot_DA:Q')

            ref_da_y_label = alt.Chart(pd.DataFrame({'plot_rMAE': [2.05], 'plot_DA': [50.8]})).mark_text(
                text='50% Random Guess Threshold',
                color='#dc2626',
                fontSize=11,
                align='left'
            ).encode(x='plot_rMAE:Q', y='plot_DA:Q')

            highlight_da = alt.selection_point(fields=['Architecture'], bind='legend')

            da_scatter = alt.Chart(da_agg).mark_circle(size=150, opacity=0.9).encode(
                x=alt.X(
                    'plot_rMAE:Q',
                    title='Magnitude Error: rMAE (Median)',
                    scale=alt.Scale(domain=[min_da_x, max_da_x])
                ),
                y=alt.Y(
                    'plot_DA:Q',
                    title='Momentum Accuracy: Directional Accuracy (%)',
                    scale=alt.Scale(domain=[min_da_y, max_da_y])
                ),
                color=alt.Color(
                    'Architecture:N',
                    title='Model Architecture',
                    scale=alt.Scale(
                        domain=['Zero-Shot', 'Baseline'],
                        range=['#1f77b4', '#ff7f0e']
                    )
                ),
                opacity=alt.condition(highlight_da, alt.value(0.95), alt.value(0.15)),
                tooltip=[
                    alt.Tooltip('Model:N', title='Model'),
                    alt.Tooltip('Architecture:N', title='Architecture'),
                    alt.Tooltip('rMAE:Q', title='Actual rMAE (Median)', format='.4f'),
                    alt.Tooltip('DA:Q', title='Actual DA (%)', format='.2f')
                ]
            ).add_params(highlight_da)

            label_pos_map_da = {
                'Bolt9m': 'bottom-left',
                'Bolt43m': 'top-left',
                'Bolt21m': 'top-right',
                'Chronos2': 'top-left',
                'TSMixer': 'bottom-right',
                'CNN': 'top-right',
                'Moirai20': 'top-left',
                'Sundial': 'top-right',
                'SeasonNaive': 'bottom-right',
                'HistoricAvg': 'top-right'
            }
            da_agg['label_pos'] = da_agg['Model'].map(label_pos_map_da).fillna('right')

            pos_styles_da = {
                'top-left': {'align': 'right', 'baseline': 'bottom', 'dx': -9, 'dy': -6},
                'top-right': {'align': 'left', 'baseline': 'bottom', 'dx': 9, 'dy': -8},
                'bottom-left': {'align': 'right', 'baseline': 'top', 'dx': -9, 'dy': 6},
                'bottom-right': {'align': 'left', 'baseline': 'top', 'dx': 9, 'dy': 8},
                'right': {'align': 'left', 'baseline': 'middle', 'dx': 11, 'dy': 0}
            }

            da_label_layers = []
            for pos_name, style in pos_styles_da.items():
                p_df = da_agg[da_agg['label_pos'] == pos_name]
                if not p_df.empty:
                    halo_layer = alt.Chart(p_df).mark_text(
                        align=style['align'],
                        baseline=style['baseline'],
                        dx=style['dx'],
                        dy=style['dy'],
                        fontSize=11,
                        stroke='white',
                        strokeWidth=3,
                        strokeJoin='round'
                    ).encode(
                        x='plot_rMAE:Q',
                        y='plot_DA:Q',
                        text='display_label:N',
                        opacity=alt.condition(highlight_da, alt.value(1.0), alt.value(0.15))
                    )
                    txt_layer = alt.Chart(p_df).mark_text(
                        align=style['align'],
                        baseline=style['baseline'],
                        dx=style['dx'],
                        dy=style['dy'],
                        fontSize=11,
                        color='black'
                    ).encode(
                        x='plot_rMAE:Q',
                        y='plot_DA:Q',
                        text='display_label:N',
                        opacity=alt.condition(highlight_da, alt.value(1.0), alt.value(0.15))
                    )
                    da_label_layers.extend([halo_layer, txt_layer])

            da_quadrant_chart = alt.layer(
                green_rect_da,
                ref_da_x,
                ref_da_y,
                ref_da_y_label,
                da_scatter,
                *da_label_layers
            ).properties(
                width='container',
                height=580,
                title=alt.TitleParams(
                    text=f"Trend Identification: rMAE vs. DA ({combo_da.upper()})",
                    anchor='middle',
                    fontSize=18,
                    offset=15,
                    orient='top'
                )
            ).interactive()

            st.altair_chart(apply_thesis_theme(da_quadrant_chart), use_container_width=True)
            st.caption(
                "**Top-Left Quadrant (< 1.0 rMAE, > 50% DA):** Pareto-superior models capturing both accurate load levels and ramp directions. &nbsp;|&nbsp; "
                "**Below 50% Line:** Negative directional tracking (e.g. quantization jitter in tokenized models). &nbsp;|&nbsp; "
                "**Zero-Shot Models** &nbsp;|&nbsp;  **Baselines**."
            )

            with st.expander("View Trend Identification Metrics Table"):
                display_da_df = da_agg.sort_values('DA', ascending=False).set_index('Model')
                st.dataframe(
                    display_da_df.style.format({
                        'rMAE': '{:.4f}',
                        'DA': '{:.2f}%'
                    }),
                    use_container_width=True
                )


        st.markdown("---")
        # --- Sector Predictability: Zero-Shot vs. Baseline Across Industry Archetypes ---
        st.subheader("Sector Predictability: Zero-Shot vs. Baseline Across Industry Archetypes")
        st.markdown(
            "Evaluates industrial load predictability across all economic sectors. Each sector is classified into "
            "one of three predictability classes (**Low**, **Medium**, or **High Predictability**) based on its empirical "
            "accuracy ceiling ($\text{Forecast Accuracy } (%) = 100% - \text{sMAPE}$). "
            "The vertical bar chart compares the aggregate performance of **Zero-Shot Models** against **Baseline Models** "
            "across the three classes (6 bars total), while the table below details the exact mapping of every industry sector."
        )

        sec_col1, sec_col2 = st.columns(2)
        size_sec = sec_col1.selectbox("Select Size (Sector Predictability)", all_sizes, key="size_sec")
        valid_combos_sec = df_metrics[df_metrics['size_category'] == size_sec]['combo_prefix'].unique().tolist()
        combo_sec = sec_col2.selectbox("Select Combination (Sector Predictability)", valid_combos_sec, key="combo_sec")

        if not df_master.empty and 'Sector_group' in df_master.columns:
            df_sec_slice = df_metrics[
                (df_metrics['size_category'] == size_sec) &
                (df_metrics['combo_prefix'] == combo_sec)
            ].replace([float('inf'), float('-inf')], pd.NA)

            if 'Sector_group' not in df_sec_slice.columns:
                df_sec_slice = df_sec_slice.merge(df_master[['id', 'Sector_group']], on='id', how='inner')

            smape_col_sec = 'horizon_smape_median_median' if 'horizon_smape_median_median' in df_sec_slice.columns else 'horizon_smape_median'
            df_sec_slice = df_sec_slice.dropna(subset=[smape_col_sec, 'Sector_group'])

            if df_sec_slice.empty:
                st.warning("No sector metric data found for the selected combination.")
            else:
                df_sec_slice['Architecture'] = df_sec_slice['model_name'].apply(
                    lambda x: 'Baseline Models' if x in BASELINE_MODELS else 'Zero-Shot Models'
                )

                sector_model_medians = df_sec_slice.groupby(['Sector_group', 'model_name', 'Architecture'])[smape_col_sec].median().reset_index()

                zs_sector_df = sector_model_medians[sector_model_medians['Architecture'] == 'Zero-Shot Models']
                best_zs_per_sector = zs_sector_df.loc[zs_sector_df.groupby('Sector_group')[smape_col_sec].idxmin()].rename(
                    columns={'model_name': 'Best Zero-Shot Model', smape_col_sec: 'zs_smape'}
                )
                best_zs_per_sector['Zero-Shot Accuracy (%)'] = 100.0 - best_zs_per_sector['zs_smape']

                base_sector_df = sector_model_medians[sector_model_medians['Architecture'] == 'Baseline Models']
                best_base_per_sector = base_sector_df.loc[base_sector_df.groupby('Sector_group')[smape_col_sec].idxmin()].rename(
                    columns={'model_name': 'Best Baseline Model', smape_col_sec: 'base_smape'}
                )
                best_base_per_sector['Baseline Accuracy (%)'] = 100.0 - best_base_per_sector['base_smape']

                if not best_zs_per_sector.empty and not best_base_per_sector.empty:
                    sector_comp_table = best_zs_per_sector[['Sector_group', 'Best Zero-Shot Model', 'Zero-Shot Accuracy (%)']].merge(
                        best_base_per_sector[['Sector_group', 'Best Baseline Model', 'Baseline Accuracy (%)']],
                        on='Sector_group'
                    )

                    sector_comp_table['Sector Accuracy (%)'] = (sector_comp_table['Zero-Shot Accuracy (%)'] + sector_comp_table['Baseline Accuracy (%)']) / 2.0
                    sector_comp_table['Zero-Shot Advantage (% pts)'] = sector_comp_table['Zero-Shot Accuracy (%)'] - sector_comp_table['Baseline Accuracy (%)']

                    sector_comp_table['Predictability Class'] = pd.qcut(
                        sector_comp_table['Sector Accuracy (%)'],
                        q=3,
                        labels=['Low Predictability', 'Medium Predictability', 'High Predictability']
                    )

                    class_bars = sector_comp_table.groupby('Predictability Class', observed=False).agg({
                        'Zero-Shot Accuracy (%)': 'median',
                        'Baseline Accuracy (%)': 'median'
                    }).reset_index()

                    df_six_bars = pd.DataFrame([
                        {
                            'Predictability Class': row['Predictability Class'],
                            'Architecture': 'Zero-Shot Models',
                            'Forecast Accuracy (%)': row['Zero-Shot Accuracy (%)']
                        } for _, row in class_bars.iterrows()
                    ] + [
                        {
                            'Predictability Class': row['Predictability Class'],
                            'Architecture': 'Baseline Models',
                            'Forecast Accuracy (%)': row['Baseline Accuracy (%)']
                        } for _, row in class_bars.iterrows()
                    ])

                    class_order = ['Low Predictability', 'Medium Predictability', 'High Predictability']
                    df_six_bars['Predictability Class'] = pd.Categorical(df_six_bars['Predictability Class'], categories=class_order, ordered=True)

                    bars = alt.Chart(df_six_bars).mark_bar(
                        cornerRadiusTopLeft=4,
                        cornerRadiusTopRight=4
                    ).encode(
                        x=alt.X(
                            'Predictability Class:O',
                            sort=class_order,
                            title='Sector Predictability Class',
                            axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15)
                        ),
                        xOffset=alt.XOffset('Architecture:N', sort=['Zero-Shot Models', 'Baseline Models']),
                        y=alt.Y(
                            'Forecast Accuracy (%):Q',
                            scale=alt.Scale(domain=[0.0, 105.0]),
                            title='Median Forecast Accuracy (100 - sMAPE %)',
                            axis=alt.Axis(labelFontSize=13, titleFontSize=15)
                        ),
                        color=alt.Color(
                            'Architecture:N',
                            title='Model Architecture',
                            scale=alt.Scale(
                                domain=['Zero-Shot Models', 'Baseline Models'],
                                range=['#1f77b4', '#ff7f0e']
                            )
                        ),
                        tooltip=[
                            alt.Tooltip('Predictability Class:N', title='Class'),
                            alt.Tooltip('Architecture:N', title='Architecture'),
                            alt.Tooltip('Forecast Accuracy (%):Q', title='Accuracy (%)', format='.2f')
                        ]
                    )

                    bar_text = alt.Chart(df_six_bars).mark_text(
                        align='center',
                        baseline='bottom',
                        dy=-6,
                        fontSize=13,
                        fontWeight='bold',
                        color='black'
                    ).encode(
                        x=alt.X('Predictability Class:O', sort=class_order),
                        xOffset=alt.XOffset('Architecture:N', sort=['Zero-Shot Models', 'Baseline Models']),
                        y=alt.Y('Forecast Accuracy (%):Q', scale=alt.Scale(domain=[0.0, 105.0])),
                        text=alt.Text('Forecast Accuracy (%):Q', format='.1f')
                    )

                    six_bar_chart = (bars + bar_text).properties(
                        width='container',
                        height=500,
                        title=alt.TitleParams(
                            text=f"Zero-Shot vs. Baseline Accuracy across Predictability Classes (Size: {size_sec.title()} | {combo_sec.upper()})",
                            anchor='middle',
                            fontSize=18,
                            offset=15,
                            orient='top'
                        )
                    )

                    st.altair_chart(apply_thesis_theme(six_bar_chart), use_container_width=True)
                    st.caption(
                        "🔵 **Zero-Shot Models** &nbsp;|&nbsp; 🟠 **Baseline Models**. "
                        "Bars display the median forecast accuracy of companies within each predictability class. "
                        "Notice the performance advantage of zero-shot models in the **Low Predictability** class."
                    )

                    st.markdown("##### Sector Predictability Classification & Best Model Mapping")
                    disp_mapping = sector_comp_table[[
                        'Sector_group',
                        'Predictability Class',
                        'Best Zero-Shot Model',
                        'Zero-Shot Accuracy (%)',
                        'Best Baseline Model',
                        'Baseline Accuracy (%)',
                        'Zero-Shot Advantage (% pts)'
                    ]].rename(columns={
                        'Sector_group': 'Economic Sector',
                        'Zero-Shot Accuracy (%)': 'Best Zero-Shot Accuracy',
                        'Baseline Accuracy (%)': 'Best Baseline Accuracy',
                        'Zero-Shot Advantage (% pts)': 'Δ Zero-Shot Advantage'
                    }).sort_values(
                        ['Predictability Class', 'Best Zero-Shot Accuracy'],
                        ascending=[True, False]
                    ).set_index('Economic Sector')

                    st.dataframe(
                        disp_mapping.style.format({
                            'Best Zero-Shot Accuracy': '{:.2f}%',
                            'Best Baseline Accuracy': '{:.2f}%',
                            'Δ Zero-Shot Advantage': '{:+.2f}% pts'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("Could not evaluate both Zero-Shot and Baseline models for this combination.")
        else:
            st.error("Sector metadata ('Sector_group') not found in master data.")


        st.markdown("---")
        # --- Seasonal & Monthly Predictability: Zero-Shot vs. Baselines ---
        st.subheader("Seasonal & Monthly Predictability: Zero-Shot vs. Baselines")
        st.markdown(
            "Evaluates whether specific calendar seasons or months are systematically more predictable than others, "
            "and how **Zero-Shot Foundation Models** compare directly against **Baseline Models** across the year."
        )

        if not df_seasonal.empty:
            seas_ctrl1, seas_ctrl2, seas_ctrl3 = st.columns([1, 1, 1.2])
            size_seas = seas_ctrl1.selectbox("Select Size (Seasonal)", all_sizes, key="size_seas")

            valid_combos_seas = df_seasonal[df_seasonal['size'] == size_seas]['combo'].unique().tolist()
            combo_seas = seas_ctrl2.selectbox("Select Combination (Seasonal)", valid_combos_seas, key="combo_seas")

            metric_choice = seas_ctrl3.radio(
                "Evaluation Metric",
                ["Absolute Error (Median MAE)", "Relative Error to SeasonNaive (Median rMAE)"],
                horizontal=True,
                key="metric_seas"
            )
            is_rmae = "rMAE" in metric_choice
            metric_col = 'median_rmae' if is_rmae else 'median_mae'
            metric_title = "Median rMAE (relative to SeasonNaive)" if is_rmae else "Median MAE (kW)"

            MONTH_NAMES_MAP = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            MONTH_TO_SEASON_MAP = {
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            }

            seas_slice = df_seasonal[(df_seasonal['size'] == size_seas) & (df_seasonal['combo'] == combo_seas)].copy()

            active_models = ['Top Zero-Shot (Chronos/Bolt/Moirai)', 'Top Baseline (SeasonNaive/TSMixer)']
            label_mapping = {
                'Top Zero-Shot (Chronos/Bolt/Moirai)': 'Zero-Shot Models',
                'Top Baseline (SeasonNaive/TSMixer)': 'Baseline Models'
            }

            seas_mode = st.radio(
                "Seasonal Representation on 12-Month Grid",
                [
                    "Seasonal Trend Line (Dashed Line connecting Season Midpoints)",
                    "Seasonal Plateau Steps (Horizontal Segments spanning each 3-Month Season)"
                ],
                horizontal=True,
                key="seas_display_mode"
            )

            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            df_m = seas_slice[
                (seas_slice['granularity'] == 'Monthly') &
                (seas_slice['model_name'].isin(active_models))
            ].copy()
            df_m['Architecture_Label'] = df_m['model_name'].map(label_mapping)
            df_m['Series_Type'] = df_m['Architecture_Label'] + ' (Monthly)'
            df_m['period_label'] = pd.Categorical(df_m['period_label'], categories=month_order, ordered=True)

            s_raw = seas_slice[
                (seas_slice['granularity'] == 'Seasonal') &
                (seas_slice['model_name'].isin(active_models))
            ].copy()

            chart_layers = []

            if "Trend Line" in seas_mode:
                season_midpoints = [
                    ('Winter', 'Jan'),
                    ('Spring', 'Apr'),
                    ('Summer', 'Jul'),
                    ('Autumn', 'Oct'),
                    ('Winter', 'Dec')
                ]
                s_rows = []
                for s_name, m_label in season_midpoints:
                    for m_name in active_models:
                        row_match = s_raw[(s_raw['model_name'] == m_name) & (s_raw['period_label'] == s_name)]
                        if not row_match.empty:
                            row = row_match.iloc[0]
                            val_fmt = f"{row[metric_col]:.2f}" if is_rmae else f"{row[metric_col]:.1f}"
                            s_rows.append({
                                'period_label': m_label,
                                'season': s_name,
                                'model_name': m_name,
                                'Architecture_Label': label_mapping[m_name],
                                'Series_Type': label_mapping[m_name] + ' (Seasonal)',
                                'label_text': f"{s_name}: {val_fmt}",
                                metric_col: row[metric_col]
                            })
                df_s = pd.DataFrame(s_rows)
                df_s['period_label'] = pd.Categorical(df_s['period_label'], categories=month_order, ordered=True)

                max_val_y = float(max(df_m[metric_col].max(), df_s[metric_col].max()) * 1.08)
                if is_rmae:
                    max_val_y = max(max_val_y, 1.15)
                    y_scale = alt.Scale(domain=[0.0, max_val_y])
                else:
                    y_scale = alt.Scale(domain=[8.0, max_val_y], zero=False)

                line_seasonal = alt.Chart(df_s).mark_line(
                    strokeDash=[6, 4],
                    strokeWidth=2.5,
                    point=alt.OverlayMarkDef(shape='diamond', filled=True, size=150)
                ).encode(
                    x=alt.X('period_label:O', sort=month_order),
                    y=alt.Y(f'{metric_col}:Q', scale=y_scale),
                    color=alt.Color('Architecture_Label:N', scale=alt.Scale(domain=['Zero-Shot Models', 'Baseline Models'], range=['#1f77b4', '#ff7f0e'])),
                    tooltip=[
                        alt.Tooltip('season:N', title='Season'),
                        alt.Tooltip('period_label:N', title='Calendar Month Position'),
                        alt.Tooltip('Architecture_Label:N', title='Architecture'),
                        alt.Tooltip(f'{metric_col}:Q', title=f'Seasonal {metric_title}', format='.2f')
                    ]
                )

                text_seasonal = alt.Chart(df_s).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-12,
                    fontSize=11,
                    fontWeight='bold'
                ).encode(
                    x=alt.X('period_label:O', sort=month_order),
                    y=alt.Y(f'{metric_col}:Q', scale=y_scale),
                    text=alt.Text('label_text:N'),
                    color=alt.Color('Architecture_Label:N', scale=alt.Scale(domain=['Zero-Shot Models', 'Baseline Models'], range=['#1f77b4', '#ff7f0e']))
                )
                chart_layers.extend([line_seasonal, text_seasonal])

            else:
                season_segments = [
                    ('Winter_1', 'Winter', ['Jan', 'Feb']),
                    ('Spring', 'Spring', ['Mar', 'Apr', 'May']),
                    ('Summer', 'Summer', ['Jun', 'Jul', 'Aug']),
                    ('Autumn', 'Autumn', ['Sep', 'Oct', 'Nov']),
                    ('Winter_2', 'Winter', ['Dec'])
                ]
                s_step_rows = []
                for seg_id, s_name, months in season_segments:
                    for m_label in months:
                        for m_name in active_models:
                            row_match = s_raw[(s_raw['model_name'] == m_name) & (s_raw['period_label'] == s_name)]
                            if not row_match.empty:
                                row = row_match.iloc[0]
                                val_fmt = f"{row[metric_col]:.2f}" if is_rmae else f"{row[metric_col]:.1f}"
                                s_step_rows.append({
                                    'period_label': m_label,
                                    'season': s_name,
                                    'seg_id': f"{seg_id}_{m_name}",
                                    'model_name': m_name,
                                    'Architecture_Label': label_mapping[m_name],
                                    'Series_Type': label_mapping[m_name] + ' (Seasonal)',
                                    'is_center': m_label in ['Jan', 'Apr', 'Jul', 'Oct'],
                                    'label_text': f"{s_name}: {val_fmt}",
                                    metric_col: row[metric_col]
                                })
                df_step = pd.DataFrame(s_step_rows)
                df_step['period_label'] = pd.Categorical(df_step['period_label'], categories=month_order, ordered=True)

                max_val_y = float(max(df_m[metric_col].max(), df_step[metric_col].max()) * 1.08)
                if is_rmae:
                    max_val_y = max(max_val_y, 1.15)
                    y_scale = alt.Scale(domain=[0.0, max_val_y])
                else:
                    y_scale = alt.Scale(domain=[8.0, max_val_y], zero=False)

                line_plateau = alt.Chart(df_step).mark_line(
                    strokeDash=[5, 3],
                    strokeWidth=3,
                    point=alt.OverlayMarkDef(shape='diamond', filled=True, size=110)
                ).encode(
                    x=alt.X('period_label:O', sort=month_order),
                    y=alt.Y(f'{metric_col}:Q', scale=y_scale),
                    detail='seg_id:N',
                    color=alt.Color('Architecture_Label:N', scale=alt.Scale(domain=['Zero-Shot Models', 'Baseline Models'], range=['#1f77b4', '#ff7f0e'])),
                    tooltip=[
                        alt.Tooltip('season:N', title='Season'),
                        alt.Tooltip('period_label:N', title='Month'),
                        alt.Tooltip('Architecture_Label:N', title='Architecture'),
                        alt.Tooltip(f'{metric_col}:Q', title=f'Seasonal {metric_title}', format='.2f')
                    ]
                )

                text_plateau = alt.Chart(df_step[df_step['is_center'] == True]).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-12,
                    fontSize=13,
                    fontWeight='bold'
                ).encode(
                    x=alt.X('period_label:O', sort=month_order),
                    y=alt.Y(f'{metric_col}:Q', scale=y_scale),
                    text=alt.Text('label_text:N'),
                    color=alt.Color('Architecture_Label:N', scale=alt.Scale(domain=['Zero-Shot Models', 'Baseline Models'], range=['#1f77b4', '#ff7f0e']))
                )
                chart_layers.extend([line_plateau, text_plateau])

            line_monthly = alt.Chart(df_m).mark_line(
                point=alt.OverlayMarkDef(shape='circle', filled=True, size=75),
                strokeWidth=2.5
            ).encode(
                x=alt.X(
                    'period_label:O',
                    sort=month_order,
                    title='Calendar Month (Mapped into 4 Seasons)',
                    axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=14)
                ),
                y=alt.Y(
                    f'{metric_col}:Q',
                    scale=y_scale,
                    title=metric_title,
                    axis=alt.Axis(labelFontSize=13, titleFontSize=15)
                ),
                color=alt.Color(
                    'Architecture_Label:N',
                    title='Model Architecture',
                    scale=alt.Scale(
                        domain=['Zero-Shot Models', 'Baseline Models'],
                        range=['#1f77b4', '#ff7f0e']
                    )
                ),
                tooltip=[
                    alt.Tooltip('period_label:N', title='Month'),
                    alt.Tooltip('season:N', title='Season'),
                    alt.Tooltip('Architecture_Label:N', title='Architecture'),
                    alt.Tooltip(f'{metric_col}:Q', title=f'Monthly {metric_title}', format='.2f')
                ]
            )
            chart_layers.insert(0, line_monthly)

            if metric_col == 'median_rmae':
                sn_ref_rule = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(
                    strokeDash=[6, 4],
                    strokeWidth=2,
                    color='#7f7f7f'
                ).encode(y='y:Q')
                chart_layers.append(sn_ref_rule)

            integrated_chart = alt.layer(*chart_layers).properties(
                width='container',
                height=480,
                title=alt.TitleParams(
                    text=f"Seasonal & Monthly Predictability: Zero-Shot vs. Baseline ({combo_seas.upper()})",
                    anchor='middle',
                    fontSize=18,
                    offset=15,
                    orient='top'
                )
            )

            st.altair_chart(apply_thesis_theme(integrated_chart), use_container_width=True)
            st.caption(
                "📈 **Solid Line + Circles (●)**: Monthly error progression across the 12 calendar months. &nbsp;|&nbsp; "
                "📊 **Dashed Line + Diamonds (◆)**: Four Seasons aggregate baseline level (Winter, Spring, Summer, Autumn).<br>"
                "🔵 **Zero-Shot Models** &nbsp;|&nbsp; 🟠 **Baseline Models**.",
                unsafe_allow_html=True
            )

            st.markdown("##### Monthly & Seasonal Predictability Breakdown")

            monthly_all = seas_slice[seas_slice['granularity'] == 'Monthly'].copy()
            macro_names = ['Zero-Shot Aggregate', 'Baseline Aggregate', 'Top Zero-Shot (Chronos/Bolt/Moirai)', 'Top Baseline (SeasonNaive/TSMixer)']
            indiv_models = monthly_all[~monthly_all['model_name'].isin(macro_names)]

            table_rows = []
            for m_num in range(1, 13):
                m_label = MONTH_NAMES_MAP[m_num]
                s_label = MONTH_TO_SEASON_MAP[m_num]

                zs_row = monthly_all[(monthly_all['period_num'] == m_num) & (monthly_all['model_name'] == active_models[0])]
                base_row = monthly_all[(monthly_all['period_num'] == m_num) & (monthly_all['model_name'] == active_models[1])]

                zs_val = zs_row[metric_col].values[0] if not zs_row.empty else np.nan
                base_val = base_row[metric_col].values[0] if not base_row.empty else np.nan

                m_indiv = indiv_models[indiv_models['period_num'] == m_num]
                best_zs_m = m_indiv[m_indiv['architecture'] == 'Zero-Shot'].sort_values(metric_col).iloc[0] if not m_indiv[m_indiv['architecture'] == 'Zero-Shot'].empty else None
                best_base_m = m_indiv[m_indiv['architecture'] == 'Baseline'].sort_values(metric_col).iloc[0] if not m_indiv[m_indiv['architecture'] == 'Baseline'].empty else None

                delta_abs = base_val - zs_val if (not pd.isna(base_val) and not pd.isna(zs_val)) else np.nan
                rel_imp_pct = ((base_val - zs_val) / base_val * 100) if (not pd.isna(base_val) and base_val != 0) else np.nan

                best_model_name = best_zs_m['model_name'] if (best_zs_m is not None and (best_base_m is None or best_zs_m[metric_col] <= best_base_m[metric_col])) else (best_base_m['model_name'] if best_base_m is not None else '-')

                table_rows.append({
                    'Month': f"{m_num:02d} - {m_label}",
                    'Season': s_label,
                    'Baseline': base_val,
                    'Zero-Shot': zs_val,
                    'Δ Advantage': delta_abs,
                    'Improvement (%)': rel_imp_pct,
                    'Best Overall Model': best_model_name
                })

            df_table = pd.DataFrame(table_rows)
            df_table['Predictability Rank'] = df_table['Baseline'].rank(ascending=True, method='min').fillna(12).astype(int)
            df_table = df_table[[
                'Month', 'Season', 'Predictability Rank',
                'Baseline', 'Zero-Shot', 'Δ Advantage', 'Improvement (%)',
                'Best Overall Model'
            ]].sort_values('Month').set_index('Month')

            format_rules = {
                'Baseline': '{:.2f}',
                'Zero-Shot': '{:.2f}',
                'Δ Advantage': '{:+.2f}',
                'Improvement (%)': '{:+.2f}%',
                'Predictability Rank': 'Rank {:d}'
            }

            st.dataframe(
                df_table.style.format(format_rules),
                use_container_width=True
            )

            st.markdown("###### Seasonal Summary (Four Seasons Aggregate)")
            seasonal_all = seas_slice[seas_slice['granularity'] == 'Seasonal'].copy()
            seas_table_rows = []
            for s_name in ['Winter', 'Spring', 'Summer', 'Autumn']:
                zs_s = seasonal_all[(seasonal_all['model_name'] == active_models[0]) & (seasonal_all['period_label'] == s_name)]
                base_s = seasonal_all[(seasonal_all['model_name'] == active_models[1]) & (seasonal_all['period_label'] == s_name)]
                z_val = zs_s[metric_col].values[0] if not zs_s.empty else np.nan
                b_val = base_s[metric_col].values[0] if not base_s.empty else np.nan
                d_val = b_val - z_val if (not pd.isna(b_val) and not pd.isna(z_val)) else np.nan
                p_val = ((b_val - z_val) / b_val * 100) if (not pd.isna(b_val) and b_val != 0) else np.nan
                seas_table_rows.append({
                    'Season': s_name,
                    'Baseline': b_val,
                    'Zero-Shot': z_val,
                    'Δ Advantage': d_val,
                    'Improvement (%)': p_val
                })
            df_seas_table = pd.DataFrame(seas_table_rows).set_index('Season')
            st.dataframe(
                df_seas_table.style.format({
                    'Baseline': '{:.2f}',
                    'Zero-Shot': '{:.2f}',
                    'Δ Advantage': '{:+.2f}',
                    'Improvement (%)': '{:+.2f}%'
                }),
                use_container_width=True
            )
        else:
            st.warning("Seasonal metrics file ('seasonal_monthly_metrics.parquet') not found.")


        st.markdown("---")
        # --- Predictability Distribution: Scale Dynamics & Leading Estimators ---
        st.subheader("Predictability Distribution: Scale Dynamics & Leading Estimators")
        st.markdown(
            "Analyzes the **empirical lower bound of predictability** (the lowest achievable sMAPE floor across all models per site) "
            "to answer: (1) Does total electricity consumption correlate with better forecasts across **Small**, **Medium**, and **Large** industrial sites? "
            "(2) Which time-series characteristics act as the most reliable estimators for the lower predictability bounds?"
        )

        pred_col1, pred_col2 = st.columns(2)
        size_pred = pred_col1.selectbox("Select Size (Predictability)", all_sizes, key="size_pred")
        valid_combos_pred = df_metrics[df_metrics['size_category'] == size_pred]['combo_prefix'].unique().tolist()
        default_combo_idx = valid_combos_pred.index('c3') if 'c3' in valid_combos_pred else 0
        combo_pred = pred_col2.selectbox("Select Combination (Predictability)", valid_combos_pred, index=default_combo_idx, key="combo_pred")

        pred_df = df_metrics[
            (df_metrics['size_category'] == size_pred) & 
            (df_metrics['combo_prefix'] == combo_pred)
        ].copy()

        smape_col = 'horizon_smape_median_median' if 'horizon_smape_median_median' in pred_df.columns else 'horizon_smape_median'
        
        lower_bound_df = pred_df.groupby('id')[smape_col].min().reset_index().rename(columns={smape_col: 'Minimum_sMAPE'})

        sn_slice = pred_df[pred_df['model_name'] == 'SeasonNaive'][['id', smape_col]].rename(columns={smape_col: 'SeasonNaive_sMAPE'})

        zs_models_combo = [m for m in pred_df['model_name'].unique() if m not in BASELINE_MODELS]
        if zs_models_combo:
            best_zs_name = pred_df[pred_df['model_name'].isin(zs_models_combo)].groupby('model_name')[smape_col].median().idxmin()
            best_zs_slice = pred_df[pred_df['model_name'] == best_zs_name][['id', smape_col]].rename(columns={smape_col: f'{best_zs_name}_sMAPE'})
        else:
            best_zs_name = 'Zero-Shot'
            best_zs_slice = pd.DataFrame(columns=['id', f'{best_zs_name}_sMAPE'])

        merged_pred = lower_bound_df.merge(sn_slice, on='id', how='left').merge(best_zs_slice, on='id', how='left')

        if not df_features.empty:
            merged_pred = merged_pred.merge(df_features, on='id', how='inner')
        if not df_master.empty:
            extra_cols = [c for c in ['repeatability_score', 'CV', 'Sector_group'] if c in df_master.columns and c not in merged_pred.columns]
            if extra_cols:
                merged_pred = merged_pred.merge(df_master[['id'] + extra_cols], on='id', how='left')

        if not merged_pred.empty and 'yearly_total_consumption_GWh' in merged_pred.columns:
            tab_scale, tab_estimators = st.tabs([
                "1. Consumption Scale Dynamics (Small, Medium, Large)",
                "2. Usage Hours Dynamics & Discretization"
            ])

            with tab_scale:
                st.markdown("##### Total Consumption vs. Forecast Predictability")
                st.markdown(
                    "All industrial sites are classified into three equal-sized tertiles based on annual electricity consumption: "
                    "**Small**, **Medium**, and **Large**. The chart below compares the theoretical **Empirical Accuracy Ceiling** "
                    rf"($100\% - \text{{sMAPE}}$) against the best performing Zero-Shot model (**{best_zs_name}**) and the best baseline (**SeasonNaive**)."
                )

                q33 = merged_pred['yearly_total_consumption_GWh'].quantile(0.333333)
                q66 = merged_pred['yearly_total_consumption_GWh'].quantile(0.666667)
                tier_labels = [
                    f"Small (≤ {q33:.2f} GWh)",
                    f"Medium ({q33:.2f}–{q66:.2f} GWh)",
                    f"Large (> {q66:.2f} GWh)"
                ]
                merged_pred['Consumption Tier'] = pd.cut(
                    merged_pred['yearly_total_consumption_GWh'],
                    bins=[-np.inf, q33, q66, np.inf],
                    labels=tier_labels
                )

                zs_col = f'{best_zs_name}_sMAPE'
                tier_summary = merged_pred.groupby('Consumption Tier', observed=False).agg(
                    Ceiling_sMAPE=('Minimum_sMAPE', 'median'),
                    SN_sMAPE=('SeasonNaive_sMAPE', 'median'),
                    ZS_sMAPE=(zs_col, 'median'),
                    Sites=('id', 'count')
                ).reset_index()

                tier_summary['Ceiling Accuracy (%)'] = 100.0 - tier_summary['Ceiling_sMAPE']
                tier_summary['Baseline Accuracy (%)'] = 100.0 - tier_summary['SN_sMAPE']
                tier_summary['Zero-Shot Accuracy (%)'] = 100.0 - tier_summary['ZS_sMAPE']
                tier_summary['Δ Advantage (% pts)'] = tier_summary['SN_sMAPE'] - tier_summary['ZS_sMAPE']

                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                for col_kpi, (_, row_t) in zip([kpi_col1, kpi_col2, kpi_col3], tier_summary.iterrows()):
                    with col_kpi:
                        st.metric(
                            label=f"{row_t['Consumption Tier']}",
                            value=f"{row_t['Zero-Shot Accuracy (%)']:.2f}% ZS Acc.",
                            delta=f"{row_t['Δ Advantage (% pts)']:+.2f}% pts vs Baseline"
                        )

                plot_rows = []
                for _, row_t in tier_summary.iterrows():
                    plot_rows.append({
                        'Consumption Tier': row_t['Consumption Tier'],
                        'Architecture': 'Predictability Ceiling',
                        'Forecast Accuracy (%)': row_t['Ceiling Accuracy (%)']
                    })
                    plot_rows.append({
                        'Consumption Tier': row_t['Consumption Tier'],
                        'Architecture': f'Best Zero-Shot ({best_zs_name})',
                        'Forecast Accuracy (%)': row_t['Zero-Shot Accuracy (%)']
                    })
                    plot_rows.append({
                        'Consumption Tier': row_t['Consumption Tier'],
                        'Architecture': 'Best Baseline (SeasonNaive)',
                        'Forecast Accuracy (%)': row_t['Baseline Accuracy (%)']
                    })
                df_tier_plot = pd.DataFrame(plot_rows)

                domain_min = float(np.floor(df_tier_plot['Forecast Accuracy (%)'].min() - 2.0))
                domain_max = float(np.ceil(df_tier_plot['Forecast Accuracy (%)'].max() + 1.5))
                tier_order = tier_labels
                arch_order = ['Predictability Ceiling', f'Best Zero-Shot ({best_zs_name})', 'Best Baseline (SeasonNaive)']

                bars = alt.Chart(df_tier_plot).mark_bar(
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4
                ).encode(
                    x=alt.X(
                        'Consumption Tier:O',
                        sort=tier_order,
                        title='Enterprise Consumption Tier (Tertiles)',
                        axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=15)
                    ),
                    xOffset=alt.XOffset('Architecture:N', sort=arch_order),
                    y=alt.Y(
                        'Forecast Accuracy (%):Q',
                        title='Forecast Accuracy (%) = 100% - sMAPE',
                        scale=alt.Scale(domain=[domain_min, domain_max], zero=False),
                        axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                    ),
                    y2=alt.datum(domain_min),
                    color=alt.Color(
                        'Architecture:N',
                        scale=alt.Scale(
                            domain=arch_order,
                            range=['#7f7f7f', '#1f77b4', '#ff7f0e']
                        ),
                        legend=alt.Legend(title='Architecture / Performance Limit', orient='top', labelFontSize=12, titleFontSize=13)
                    ),
                    tooltip=[
                        alt.Tooltip('Consumption Tier:N', title='Tier'),
                        alt.Tooltip('Architecture:N', title='Model / Ceiling'),
                        alt.Tooltip('Forecast Accuracy (%):Q', format='.2f')
                    ]
                )

                bar_text = alt.Chart(df_tier_plot).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,
                    fontSize=11,
                    fontWeight='bold'
                ).encode(
                    x=alt.X('Consumption Tier:O', sort=tier_order),
                    xOffset=alt.XOffset('Architecture:N', sort=arch_order),
                    y=alt.Y('Forecast Accuracy (%):Q'),
                    text=alt.Text('Forecast Accuracy (%):Q', format='.2f'),
                    color=alt.value('black')
                )

                chart_tier = (bars + bar_text).properties(
                    width='container',
                    height=440
                )

                st.altair_chart(apply_thesis_theme(chart_tier), use_container_width=True)

                st.markdown("###### Reproducible Consumption Tier Performance Breakdown")
                st.caption(
                    "Proves exact empirical calculations across enterprise size tiers. "
                    "Formulas: Empirical Floor = median(min_m sMAPE); Accuracy Ceiling = 100% - Empirical Floor; "
                    "Zero-Shot Advantage (Δ % pts) = SeasonNaive sMAPE - Best Zero-Shot sMAPE."
                )

                tier_summary['Annual Volume'] = [
                    f"≤ {q33:.3f} GWh",
                    f"{q33:.3f} – {q66:.3f} GWh",
                    f"> {q66:.3f} GWh"
                ]

                disp_tier_table = tier_summary[[
                    'Consumption Tier', 'Annual Volume', 'Sites', 
                    'Ceiling_sMAPE', 'Ceiling Accuracy (%)',
                    'SN_sMAPE', 'ZS_sMAPE', 'Δ Advantage (% pts)'
                ]].copy()

                disp_tier_table.rename(columns={
                    'Ceiling_sMAPE': 'Empirical Floor (sMAPE %)',
                    'Ceiling Accuracy (%)': 'Accuracy Ceiling (%)',
                    'SN_sMAPE': 'Best Baseline (SeasonNaive %)',
                    'ZS_sMAPE': f'Best Zero-Shot ({best_zs_name} %)',
                    'Δ Advantage (% pts)': 'Zero-Shot Advantage (Δ % pts)'
                }, inplace=True)

                st.dataframe(
                    disp_tier_table.style.format({
                        'Sites': '{:,d}',
                        'Empirical Floor (sMAPE %)': '{:.2f}%',
                        'Accuracy Ceiling (%)': '{:.2f}%',
                        'Best Baseline (SeasonNaive %)': '{:.2f}%',
                        f'Best Zero-Shot ({best_zs_name} %)': '{:.2f}%',
                        'Zero-Shot Advantage (Δ % pts)': '{:+.2f}% pts'
                    }),
                    use_container_width=True
                )

            with tab_estimators:
                st.markdown("##### Leading Predictability Estimator: Usage Hours")
                st.markdown(
                    "Statistical correlation and quantile progression of **Usage Hours** against the **Empirical Lower Bound (Min sMAPE)**. "
                    "Usage hours represents annual full load hours (Total kWh / Peak kW), quantifying baseload continuity versus peak-driven demand."
                )

                if 'usage_hours_h' in merged_pred.columns:
                    v = merged_pred.dropna(subset=['Minimum_sMAPE', 'usage_hours_h'])
                    r_pearson = v['Minimum_sMAPE'].corr(v['usage_hours_h'])
                    rho_spearman = v['Minimum_sMAPE'].corr(v['usage_hours_h'], method='spearman')

                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric(
                            label="Spearman Rank Correlation (ρ)",
                            value=f"{rho_spearman:+.4f}",
                            help="Rank-order monotonic correlation against empirical min sMAPE floor."
                        )
                    with m_col2:
                        st.metric(
                            label="Pearson Correlation (r)",
                            value=f"{r_pearson:+.4f}",
                            help="Linear correlation against empirical min sMAPE floor."
                        )
                    with m_col3:
                        st.metric(
                            label="Predictability Impact",
                            value="Strong Predictability Booster",
                            delta="Lower Error Floor",
                            delta_color="normal"
                        )

                    st.markdown("---")
                    st.markdown("##### Discretized Usage Hours Quantile Explorer")
                    st.markdown(
                        "Visualizes the progression of the **Empirical Lower Bound (Floor)** alongside **Best Zero-Shot** and **Baseline** models "
                        "across discretized quantiles of Usage Hours."
                    )

                    num_bins_choice = st.selectbox(
                        "Select Bin Granularity",
                        ["5 Bins (Quintiles)", "10 Bins (Deciles)"],
                        index=0,
                        key="num_bins_disc"
                    )
                    n_bins = 5 if "5" in num_bins_choice else 10

                    valid_disc = merged_pred.dropna(subset=['usage_hours_h', 'Minimum_sMAPE', 'SeasonNaive_sMAPE', zs_col]).copy()

                    if not valid_disc.empty:
                        valid_disc['bin'] = pd.qcut(valid_disc['usage_hours_h'], q=n_bins, duplicates='drop')
                        bin_agg = valid_disc.groupby('bin', observed=False).agg(
                            Floor_sMAPE=('Minimum_sMAPE', 'median'),
                            SN_sMAPE=('SeasonNaive_sMAPE', 'median'),
                            ZS_sMAPE=(zs_col, 'median'),
                            Sites=('id', 'count')
                        ).reset_index()

                        bin_agg['Bin_Label'] = bin_agg['bin'].apply(lambda x: f"({x.left:.1f}, {x.right:.1f}] h")
                        bin_agg['Δ Advantage (% pts)'] = bin_agg['SN_sMAPE'] - bin_agg['ZS_sMAPE']
                        bin_agg['Order'] = range(len(bin_agg))

                        df_disc_lines = bin_agg.melt(
                            id_vars=['Bin_Label', 'Order'],
                            value_vars=['Floor_sMAPE', 'SN_sMAPE', 'ZS_sMAPE'],
                            var_name='Series',
                            value_name='sMAPE (%)'
                        )
                        series_name_map = {
                            'Floor_sMAPE': 'Empirical Lower Bound (Floor)',
                            'ZS_sMAPE': f'Best Zero-Shot ({best_zs_name})',
                            'SN_sMAPE': 'Best Baseline (SeasonNaive)'
                        }
                        df_disc_lines['Model / Bound'] = df_disc_lines['Series'].map(series_name_map)

                        chart_disc_lines = alt.Chart(df_disc_lines).mark_line(point=True, strokeWidth=3).encode(
                            x=alt.X(
                                'Bin_Label:O',
                                title="Usage Hours Quantile Bins",
                                sort=alt.EncodingSortField(field='Order', order='ascending'),
                                axis=alt.Axis(labelAngle=-15, labelFontSize=12, titleFontSize=14)
                            ),
                            y=alt.Y(
                                'sMAPE (%):Q',
                                title='sMAPE (%)',
                                scale=alt.Scale(zero=False),
                                axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                            ),
                            color=alt.Color(
                                'Model / Bound:N',
                                scale=alt.Scale(
                                    domain=['Empirical Lower Bound (Floor)', f'Best Zero-Shot ({best_zs_name})', 'Best Baseline (SeasonNaive)'],
                                    range=['#7f7f7f', '#1f77b4', '#ff7f0e']
                                ),
                                legend=alt.Legend(title='Performance Floor & Models', orient='top', labelFontSize=12, titleFontSize=13)
                            ),
                            tooltip=['Bin_Label:N', 'Model / Bound:N', alt.Tooltip('sMAPE (%):Q', format='.2f')]
                        ).properties(height=340, width='container')

                        chart_disc_bar = alt.Chart(bin_agg).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color='#1f77b4').encode(
                            x=alt.X(
                                'Bin_Label:O',
                                title=None,
                                sort=alt.EncodingSortField(field='Order', order='ascending'),
                                axis=alt.Axis(labelAngle=-15, labelFontSize=12)
                            ),
                            y=alt.Y(
                                'Δ Advantage (% pts):Q',
                                title='Δ ZS Advantage (% pts)',
                                axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                            ),
                            tooltip=['Bin_Label:N', alt.Tooltip('Δ Advantage (% pts):Q', format='+.2f'), alt.Tooltip('Sites:Q', format=',d')]
                        ).properties(height=180, width='container')

                        st.altair_chart(apply_thesis_theme(chart_disc_lines), use_container_width=True)
                        st.altair_chart(apply_thesis_theme(chart_disc_bar), use_container_width=True)

                        st.markdown("###### Quantile Bin Performance Breakdown")
                        disp_bin_table = bin_agg[['Bin_Label', 'Sites', 'Floor_sMAPE', 'SN_sMAPE', 'ZS_sMAPE', 'Δ Advantage (% pts)']].copy()
                        disp_bin_table.rename(columns={
                            'Bin_Label': 'Usage Hours Range',
                            'Floor_sMAPE': 'Empirical Floor (sMAPE %)',
                            'SN_sMAPE': 'SeasonNaive (sMAPE %)',
                            'ZS_sMAPE': f'{best_zs_name} (sMAPE %)',
                            'Δ Advantage (% pts)': 'Zero-Shot Advantage (Δ % pts)'
                        }, inplace=True)
                        st.dataframe(
                            disp_bin_table.style.format({
                                'Sites': '{:,d}',
                                'Empirical Floor (sMAPE %)': '{:.2f}%',
                                'SeasonNaive (sMAPE %)': '{:.2f}%',
                                f'{best_zs_name} (sMAPE %)': '{:.2f}%',
                                'Zero-Shot Advantage (Δ % pts)': '{:+.2f}% pts'
                            }),
                            use_container_width=True
                        )
                else:
                    st.warning("Usage Hours column ('usage_hours_h') not found in features.")
        else:
            st.warning("Load profile features not available or no overlapping IDs found.")

        st.markdown("---")
        # --- Impact of the Portfolio Effect: Multi-Site Aggregation ---
        st.subheader("Impact of the Portfolio Effect: Multi-Site Aggregation")
        st.markdown(
            "Evaluates how aggregating individual industrial load profiles into collective portfolios affects forecast predictability. "
            "Under the portfolio effect (Law of Large Numbers), uncorrelated stochastic demand shocks cancel out, lowering relative forecast error. "
            "This experiment evaluates portfolio scaling across nine orders of magnitude from **K = 1 to 5,000 industrial sites**."
        )

        if not df_portfolio.empty:
            grid_levels = {
                1: "Single Industrial Site",
                10: "Local Microgrid / Energy Community",
                20: "Industrial Park Feeder",
                50: "Distribution Substation Level",
                100: "Municipal Utility Balancing Group",
                200: "Regional Distribution Portfolio",
                500: "Large Regional Aggregator",
                1000: "Distribution System Operator (DSO)",
                2000: "Transmission Feeder Level",
                5000: "Nationwide Industrial Grid Portfolio"
            }
            k_order = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            label_order = [f"{k:,}" for k in k_order]

            df_port_view = df_portfolio.copy()
            df_port_view['Operational Scale'] = df_port_view['Portfolio_Size_K'].map(grid_levels).fillna("Aggregated Portfolio")
            df_port_view['K_Label'] = df_port_view['Portfolio_Size_K'].apply(lambda x: f"{x:,}")

            df_plot_port = df_port_view.melt(
                id_vars=['Portfolio_Size_K', 'K_Label'],
                value_vars=['Chronos2_sMAPE', 'SeasonNaive_sMAPE', 'Theoretical_1_sqrt_K'],
                var_name='Series',
                value_name='sMAPE (%)'
            )
            name_map = {
                'Chronos2_sMAPE': 'Zero-Shot Model (Chronos2)',
                'SeasonNaive_sMAPE': 'Baseline (SeasonNaive)',
                'Theoretical_1_sqrt_K': 'Theoretical Uncorrelated Benchmark'
            }
            df_plot_port['Model / Benchmark'] = df_plot_port['Series'].map(name_map)

            color_scale = alt.Scale(
                domain=[
                    'Zero-Shot Model (Chronos2)',
                    'Baseline (SeasonNaive)',
                    'Theoretical Uncorrelated Benchmark'
                ],
                range=['#1f77b4', '#ff7f0e', '#7f7f7f']
            )

            dash_scale = alt.Scale(
                domain=[
                    'Zero-Shot Model (Chronos2)',
                    'Baseline (SeasonNaive)',
                    'Theoretical Uncorrelated Benchmark'
                ],
                range=[[0], [0], [6, 4]]
            )

            chart_port = alt.Chart(df_plot_port).mark_line(
                point=alt.OverlayMarkDef(filled=True, size=55),
                strokeWidth=2.8
            ).encode(
                x=alt.X(
                    'K_Label:O',
                    sort=label_order,
                    title='Portfolio Size K (Aggregated Industrial Sites)',
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=12,
                        titleFontSize=14
                    )
                ),
                y=alt.Y(
                    'sMAPE (%):Q',
                    title='Portfolio Forecast Error (sMAPE %)',
                    scale=alt.Scale(zero=False, domain=[0, 20]),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                ),
                color=alt.Color(
                    'Model / Benchmark:N',
                    scale=color_scale,
                    legend=alt.Legend(
                        title='Model / Benchmark',
                        orient='top',
                        labelFontSize=12,
                        titleFontSize=13
                    )
                ),
                strokeDash=alt.StrokeDash('Model / Benchmark:N', scale=dash_scale, legend=None),
                tooltip=[
                    alt.Tooltip('Portfolio_Size_K:Q', title='Portfolio Size (K)', format=',d'),
                    alt.Tooltip('Model / Benchmark:N', title='Series'),
                    alt.Tooltip('sMAPE (%):Q', title='sMAPE (%)', format='.2f')
                ]
            ).properties(
                width='container',
                height=480,
                title=alt.TitleParams(
                    text="Portfolio Predictability Multi-Site Aggregation (K = 1 to 5,000)",
                    anchor='middle',
                    fontSize=18,
                    offset=15
                )
            )

            st.altair_chart(apply_thesis_theme(chart_port), use_container_width=True)

            st.markdown("###### Empirical Portfolio Aggregation Scaling Breakdown")
            st.caption(
                "Evaluates forecast error progression across increasing portfolio scales. "
                "Zero-Shot Advantage (Δ % pts) = SeasonNaive sMAPE - Chronos2 sMAPE; "
                "Relative Error Reduction (%) = (SeasonNaive - Chronos2) / SeasonNaive * 100; "
                "Theoretical Benchmark = sMAPE(K=1) / sqrt(K)."
            )

            df_port_view['ZeroShot_Advantage_pts'] = df_port_view['SeasonNaive_sMAPE'] - df_port_view['Chronos2_sMAPE']
            df_port_view['Relative_Error_Reduction_pct'] = (
                (df_port_view['SeasonNaive_sMAPE'] - df_port_view['Chronos2_sMAPE']) / df_port_view['SeasonNaive_sMAPE']
            ) * 100

            disp_port_table = df_port_view[[
                'Portfolio_Size_K', 'Operational Scale',
                'Chronos2_sMAPE', 'SeasonNaive_sMAPE',
                'ZeroShot_Advantage_pts', 'Relative_Error_Reduction_pct',
                'Theoretical_1_sqrt_K'
            ]].copy()

            disp_port_table.rename(columns={
                'Portfolio_Size_K': 'Portfolio Size (K)',
                'Chronos2_sMAPE': 'Chronos2 (sMAPE %)',
                'SeasonNaive_sMAPE': 'SeasonNaive (sMAPE %)',
                'ZeroShot_Advantage_pts': 'Zero-Shot Advantage (Δ % pts)',
                'Relative_Error_Reduction_pct': 'Relative Error Reduction (%)',
                'Theoretical_1_sqrt_K': 'Theoretical Benchmark (1/√K %)'
            }, inplace=True)

            st.dataframe(
                disp_port_table.style.format({
                    'Portfolio Size (K)': '{:,d}',
                    'Chronos2 (sMAPE %)': '{:.2f}%',
                    'SeasonNaive (sMAPE %)': '{:.2f}%',
                    'Zero-Shot Advantage (Δ % pts)': '{:+.2f}% pts',
                    'Relative Error Reduction (%)': '{:+.2f}%',
                    'Theoretical Benchmark (1/√K %)': '{:.2f}%'
                }),
                use_container_width=True
            )
        else:
            st.warning("Portfolio metrics file ('portfolio_aggregation_metrics.parquet') not found.")
                
    else:
        st.warning("Master metrics parquet file is empty or not generated yet.")
else:
    st.error("Invalid directory path. Please provide the root directory containing 'master_metrics.parquet'.")