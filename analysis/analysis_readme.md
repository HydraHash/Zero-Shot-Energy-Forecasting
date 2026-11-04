# Zero-Shot Forecasting Results Analysis Implementation
This folder contains implementations for the analysis of the generated forecast files.
All core metrics are calculated based on the results and saved into a single parquet file which is loaded by the streamlit dashboard.

Calculation of the Coefficient of Variation to add to masterdata file: calculate_cv.py
Streamlit interactive dashboard: dashboard.py (command: streamlit run dashboard.py)
Calculation of core metrics: export_metrics.py
Overview of Results Folder and correct file naming: print_tree.py

## Virtual Environment and Python Version
Python-Version used: Python 3.13.12
All required packages are contained in requirements.txt file.
