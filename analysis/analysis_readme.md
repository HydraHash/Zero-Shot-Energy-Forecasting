# Results Analysis & Evaluation Dashboard

This directory contains scripts for computing forecast metrics and running the interactive evaluation dashboard.

## Contained Files

* **`dashboard.py`**: Interactive Streamlit dashboard for evaluating forecast accuracy across models, horizons, repeatability tiers, sectors, and portfolio scales.
* **`export_metrics.py`**: Computes and exports core error metrics ($r\text{MAE}$, $\text{RMSE}$, $\text{sMAPE}$, $\text{DA}$) from raw predictions into `master_metrics.parquet`.
* **`generate_seasonal_metrics.py`**: Computes monthly and seasonal metrics into `seasonal_monthly_metrics.parquet`.
* **`calculate_cv.py`**: Computes profile Coefficient of Variation ($\text{CV}$) and daily repeatability scores.
* **`evaluation_metrics_v1.py`**: Mathematical definitions and formulas for all evaluation metrics.
* **`comparison_report.py`**: Generates cross-model benchmark comparison summaries.
* **`analyze_format.py` & `print_tree.py`**: Utilities for verifying output schemas and directory structure.
* **`inference_time_overview.txt`**: Tabulated model inference runtimes across forecast combinations.

## Running the Dashboard

```bash
# 1. Activate environment (optional)
.\venv\Scripts\Activate.ps1

# 2. Start Streamlit dashboard
streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
