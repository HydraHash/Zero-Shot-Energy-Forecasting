# Chronos-2, Chronos-Bolt Zero-Shot Forecasting Implementation
This folder contains implementations for the Chronos-2 and Chronos-Bolt models. 
Since all models (Chronos-2, Chronos-Bolt 9m, 21m and 43m) were published by the same organization, all four models are implemented in this shared folder.
Based on this, the implementations are very similar.


Chronos-2 Implementation: zs_forecast_base.py
Chronos-Bolt 9m, 21m, 43m Implementation: zs_forecast_bolt.py
This implementation runs all three model parameters sizes in one pipeline.

## Virtual Environment and Python Version
Python-Version used: Python 3.13.12
All required packages are contained in requirements.txt file.

## Link to GitHub Repository
[text](https://github.com/amazon-science/chronos-forecasting)

## Adjustments prior to execution
### zs_forecast_base.py
    - Set Output and Data path 
    - Configure desired batch size 
    - Set name of log file 

### zs_forecast_bolt.py
    - Set Output Base Directory Path (run_modelsize())
    - Set Data Path
    - Configure desired batch size
    - Set name of log file