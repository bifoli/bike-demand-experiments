# Probabilistic Bike-Sharing Demand Forecasting (Reproducible Experiments)

This codebase reproduces the experiments described in the manuscript:
**“Probabilistic Bike-Sharing Demand Forecasting under Changing Weather and Seasonal Regimes with Transformer-Based Models”**.

It trains tree baselines (RF, XGBoost) and deep models (N-BEATS, Transformer encoder, Informer-lite, TFT),
produces point + quantile forecasts (q={0.1,0.5,0.9}), and computes:
- MAE / RMSE / MAPE on the median (P50)
- Empirical 80% interval coverage + width (P10–P90)
- Robustness metrics by regime (Normal / Holiday / Extreme weather)
- Figures (interval case study, hourly MAE, TFT feature importance)

## 1) Environment
Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

## 2) Data
Place the datasets in `data/`:

https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

```bash
python run_experiments.py 

(Downloads from the UCI repository.)

## 3) Run all experiments
```bash
python run_experiments.py --output_dir outputs --seed 42
```

This will write:
- `outputs/results.json` (tables)
- `outputs/fig1_interval.png`
- `outputs/fig2_hourly_mae.png`
- `outputs/fig3_tft_importance.png`

## 4) Notes on reproducibility
- Fixed random seed (default 42).
- CPU execution is supported; GPU is optional.
- Neural models use early stopping on validation pinball loss.

## 5) License
This code is provided for research/review purposes. The datasets are distributed under their respective licenses.
