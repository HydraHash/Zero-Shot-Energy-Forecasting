# Zero-Shot Industrial Energy Demand Forecasting

This repository contains the experimental pipeline, model implementations, evaluation suite, and thesis writeup for the large-scale analysis of industrial electricity load predictability using Zero-Shot Time Series Foundation Models (TSFMs).

The benchmark systematically evaluates 18 zero-shot foundation model variants against classical statistical and supervised deep learning baselines across 5,359 commercial and industrial (C&I) load profiles at 15-minute resolution across multiple temporal horizon configurations ($c_1\text{--}c_4$).

---

## Repository Structure & Model Scripts

```text
├── analysis/              # Metric aggregation, diagnostic tools, and Streamlit evaluation dashboard
│   ├── dashboard.py                       # Interactive evaluation dashboard
│   ├── export_metrics.py                  # Computes master_metrics.parquet from prediction batches
│   └── generate_seasonal_metrics.py       # Computes seasonal_monthly_metrics.parquet
│
├── baselines/             # Statistical and supervised deep learning baselines
│   ├── statistical_models/vectorized_predictions.py    # Seasonal Naïve & Historical Average (Py 3.13)
│   └── model_tsmixer/tsmixer/                          # Supervised CNN & TSMixer (Py 3.12)
│       ├── cnn_prediction_batch.py
│       └── tsmixer_prediction_batch.py
│
├── model_Chronos/         # Amazon Chronos-2 & Chronos-Bolt (Py 3.13)
│   ├── zs_forecast_base.py                # Chronos-2 universal group-attention model
│   └── zs_forecast_bolt.py                # Chronos-Bolt (9M, 21M, 43M) patched encoder-decoder
│
├── model_Kairos/          # IBM Kairos (Py 3.13)
│   └── zs_forecast_kairos.py              # Kairos 10M, 23M, 50M parameter checkpoints
│
├── model_Lag_Llama/       # Morgan Stanley / Mila Lag-Llama (Py 3.13)
│   └── lag-llama/zs_forecast.py           # Probabilistic decoder-only foundation model
│
├── model_Moirai2/         # Salesforce Moirai / Uni2TS (Py 3.13)
│   └── uni2ts/zs_forecast.py              # Moirai 1.0, 1.1, and 2.0 masked autoencoder
│
├── model_Moment1/         # AutonLab Moment-1 (Py 3.13)
│   └── zs_forecast_moment.py              # Multi-task foundation model
│
├── model_TimeLLM/         # Time-LLM (Py 3.13)
│   └── Time-LLM/run_timellm_tsz.py        # Reprogrammed LLM true zero-shot inference
│
├── model_Timer_Sundial/   # THUML Large Time Series Models (Py 3.10)
│   ├── zs_forecast_sundial.py             # Sundial-Base (128M) causal LM
│   └── zs_forecast_timer.py               # Timer-Base autoregressive foundation model
│
├── model_TimesFM/         # Google Research TimesFM (Py 3.13)
│   ├── timesfm/zs_forecast.py             # TimesFM 2.5 (200M)
│   └── timesfm/zs_forecast_v3.py          # TimesFM 3.0 (330M)
│
├── model_Tirex/           # NX-AI TiRex (Py 3.13)
│   └── forecast_zs.py                     # xLSTM recurrent state-space foundation model
│
└── writeup/               # Master's thesis LaTeX source code, chapters, and bibliography
```

---

## Experimental Workflow

### 1. Model Inference
Each model script runs in an isolated virtual environment (`requirements.txt` included in each directory).

**Required adjustments prior to execution** (in `if __name__ == "__main__":`):
* `DATA_PATH`: Path to the continuous 15-minute load profile data (`.tsv` or `.parquet`)
* `OUTPUT_BASE_DIR`: Target directory where batched predictions are saved
* `BATCH_SIZE`: Number of parallel companies processed per inference batch (typically `250`–`500`)

**Temporal Combinations ($c_1\text{--}c_4$)**:
* **$c_1$**: 1-day context $\to$ 1-day horizon (step: 1 day)
* **$c_2$**: 7-day context $\to$ 7-day horizon (step: 7 days)
* **$c_3$**: 7-day context $\to$ 1-day horizon (step: 1 day)
* **$c_4$**: 14-day context $\to$ 1-day horizon (step: 1 day)

### 2. Metrics Aggregation
Once inference batches are generated, run the metric extraction pipelines to generate the unified datasets:
```bash
cd analysis
python export_metrics.py            # Generates master_metrics.parquet
python generate_seasonal_metrics.py # Generates seasonal_monthly_metrics.parquet
```

### 3. Interactive Evaluation Dashboard
Launch Streamlit to inspect model benchmarks, predictability distributions, and portfolio scaling:
```bash
cd analysis
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser.