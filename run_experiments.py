from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

from src.data_utils import download_data, prepare_uci, prepare_seoul, make_tree_supervised, make_seq_supervised
from src.metrics import mae, rmse, mape, interval_coverage, interval_width
from src.models_tree import RFQuantile, XGBQuantile
from src.models_deep import (
    set_seed, TrainConfig,
    NBeatsLite, TransformerEncoderModel, InformerLiteModel, TFTLite,
    train_model, predict_model, tft_feature_importance
)
from src.plotting import plot_interval_case_study, plot_hourly_mae, plot_feature_importance

QUANTILES = (0.1, 0.5, 0.9)
LOOKBACK = 24

def inv_transform(pred_z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    pred_log = pred_z * sigma + mu
    pred = np.expm1(pred_log)
    return np.maximum(pred, 0.0)

def eval_forecast(y_true: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, q50),
        "RMSE": rmse(y_true, q50),
        "MAPE": mape(y_true, q50),
        "P80_Coverage": interval_coverage(y_true, q10, q90),
        "P80_Width": interval_width(q10, q90),
    }

def run_dataset(name: str, data, out_dir: str, seed: int) -> Dict:
    set_seed(seed)

    # Tree supervised
    Xtr_tree, ytr = make_tree_supervised(data.ylogz_train, data.cov_train, lookback=LOOKBACK)
    Xva_tree, yva = make_tree_supervised(data.ylogz_val, data.cov_val, lookback=LOOKBACK)
    Xte_tree, yte = make_tree_supervised(data.ylogz_test, data.cov_test, lookback=LOOKBACK)

    # Seq supervised
    Xtr_seq, ytr_s = make_seq_supervised(data.ylogz_train, data.cov_train, lookback=LOOKBACK)
    Xva_seq, yva_s = make_seq_supervised(data.ylogz_val, data.cov_val, lookback=LOOKBACK)
    Xte_seq, yte_s = make_seq_supervised(data.ylogz_test, data.cov_test, lookback=LOOKBACK)

    assert np.allclose(ytr, ytr_s)
    assert np.allclose(yva, yva_s)
    assert np.allclose(yte, yte_s)

    # Align true y in count scale for evaluation (skip first LOOKBACK in each split)
    y_true_test = data.y_test[LOOKBACK:]
    dt_test = data.dt_test[LOOKBACK:]

    results = {}

    # 1) RF
    rf = RFQuantile(n_estimators=200, max_depth=18, random_state=seed)
    rf.fit(Xtr_tree, ytr)
    rf_pred = rf.predict(Xte_tree)
    rf_q10 = inv_transform(rf_pred.q10, data.ylog_mu, data.ylog_sigma)
    rf_q50 = inv_transform(rf_pred.q50, data.ylog_mu, data.ylog_sigma)
    rf_q90 = inv_transform(rf_pred.q90, data.ylog_mu, data.ylog_sigma)
    results["RF-Quantile"] = eval_forecast(y_true_test, rf_q10, rf_q50, rf_q90)

    # 2) XGB
    xgb_params = dict(
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    xg = XGBQuantile(params=xgb_params, random_state=seed)
    xg.fit(Xtr_tree, ytr)
    xg_pred = xg.predict(Xte_tree)
    xg_q10 = inv_transform(xg_pred.q10, data.ylog_mu, data.ylog_sigma)
    xg_q50 = inv_transform(xg_pred.q50, data.ylog_mu, data.ylog_sigma)
    xg_q90 = inv_transform(xg_pred.q90, data.ylog_mu, data.ylog_sigma)
    results["XGBoost-Quantile"] = eval_forecast(y_true_test, xg_q10, xg_q50, xg_q90)

    # Deep training config
    device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") and __import__("torch").cuda.is_available()) else "cpu"

    # 3) N-BEATS
    nb = NBeatsLite(in_dim=Xtr_seq.shape[1] * Xtr_seq.shape[2], hidden=256, n_layers=4)
    nb_cfg = TrainConfig(lr=1e-3, batch_size=256, max_epochs=8, patience=2, device=device)
    nb = train_model(nb, Xtr_seq, ytr, Xva_seq, yva, nb_cfg)
    nb_pred = predict_model(nb, Xte_seq, device=device)
    nb_q10 = inv_transform(nb_pred[:,0], data.ylog_mu, data.ylog_sigma)
    nb_q50 = inv_transform(nb_pred[:,1], data.ylog_mu, data.ylog_sigma)
    nb_q90 = inv_transform(nb_pred[:,2], data.ylog_mu, data.ylog_sigma)
    results["N-BEATS"] = eval_forecast(y_true_test, nb_q10, nb_q50, nb_q90)

    # 4) Transformer
    trf = TransformerEncoderModel(feat_dim=Xtr_seq.shape[2], d_model=64, nhead=4, num_layers=2)
    trf_cfg = TrainConfig(lr=1e-3, batch_size=256, max_epochs=6, patience=2, device=device)
    trf = train_model(trf, Xtr_seq, ytr, Xva_seq, yva, trf_cfg)
    trf_pred = predict_model(trf, Xte_seq, device=device)
    trf_q10 = inv_transform(trf_pred[:,0], data.ylog_mu, data.ylog_sigma)
    trf_q50 = inv_transform(trf_pred[:,1], data.ylog_mu, data.ylog_sigma)
    trf_q90 = inv_transform(trf_pred[:,2], data.ylog_mu, data.ylog_sigma)
    results["Transformer"] = eval_forecast(y_true_test, trf_q10, trf_q50, trf_q90)

    # 5) Informer-lite
    inf = InformerLiteModel(feat_dim=Xtr_seq.shape[2], d_model=48, nhead=3, num_layers=1)
    inf_cfg = TrainConfig(lr=1e-3, batch_size=256, max_epochs=6, patience=2, device=device)
    inf = train_model(inf, Xtr_seq, ytr, Xva_seq, yva, inf_cfg)
    inf_pred = predict_model(inf, Xte_seq, device=device)
    inf_q10 = inv_transform(inf_pred[:,0], data.ylog_mu, data.ylog_sigma)
    inf_q50 = inv_transform(inf_pred[:,1], data.ylog_mu, data.ylog_sigma)
    inf_q90 = inv_transform(inf_pred[:,2], data.ylog_mu, data.ylog_sigma)
    results["Informer-lite"] = eval_forecast(y_true_test, inf_q10, inf_q50, inf_q90)

    # 6) TFT
    tft = TFTLite(feat_dim=Xtr_seq.shape[2], d_model=64, lstm_hidden=64, nhead=4)
    tft_cfg = TrainConfig(lr=1e-3, batch_size=256, max_epochs=6, patience=2, device=device)
    tft = train_model(tft, Xtr_seq, ytr, Xva_seq, yva, tft_cfg)
    tft_pred = predict_model(tft, Xte_seq, device=device)
    tft_q10 = inv_transform(tft_pred[:,0], data.ylog_mu, data.ylog_sigma)
    tft_q50 = inv_transform(tft_pred[:,1], data.ylog_mu, data.ylog_sigma)
    tft_q90 = inv_transform(tft_pred[:,2], data.ylog_mu, data.ylog_sigma)
    results["TFT"] = eval_forecast(y_true_test, tft_q10, tft_q50, tft_q90)

    # Save figures only for UCI (to match manuscript)
    if name.lower() == "uci":
        # Fig1: interval case study
        plot_interval_case_study(
            dt=dt_test, y_true=y_true_test, q10=tft_q10, q50=tft_q50, q90=tft_q90,
            start_ts="2012-07-01", days=7,
            out_path=os.path.join(out_dir, "fig1_interval.png"),
            title="Figure 1. TFT median forecast with 80% interval (UCI test week)"
        )

        # Fig2: hourly MAE
        hours = data.df_test["hr"].to_numpy()[LOOKBACK:]
        model_err = {
            "RF": np.abs(y_true_test - rf_q50),
            "XGB": np.abs(y_true_test - xg_q50),
            "N-BEATS": np.abs(y_true_test - nb_q50),
            "Transformer": np.abs(y_true_test - trf_q50),
            "Informer-lite": np.abs(y_true_test - inf_q50),
            "TFT": np.abs(y_true_test - tft_q50),
        }
        plot_hourly_mae(hours=hours, model_to_abs_err=model_err, out_path=os.path.join(out_dir, "fig2_hourly_mae.png"),
                        title="Figure 2. Hourly MAE on UCI test year")

        # Fig3: TFT feature importance
        w = tft_feature_importance(tft, Xte_seq, device=device)
        plot_feature_importance(
            names=["demand","obs"] + data.cov_names,
            weights=w,
            top_k=12,
            out_path=os.path.join(out_dir, "fig3_tft_importance.png"),
            title="Figure 3. TFT feature importance (mean gate weight)"
        )

    return results, {
        "dt_test": dt_test,
        "y_true_test": y_true_test,
        "xgb_q": (xg_q10, xg_q50, xg_q90),
        "tft_q": (tft_q10, tft_q50, tft_q90),
    }

def regime_table(dataset_name: str, data, cache: Dict) -> Dict:
    """Compute Table 3 style MAE + P80 coverage by regime for XGB and TFT."""
    y = cache["y_true_test"]
    xg_lo, xg_mid, xg_hi = cache["xgb_q"]
    tft_lo, tft_mid, tft_hi = cache["tft_q"]

    df_test = data.df_test.iloc[LOOKBACK:].copy()

    if dataset_name.lower() == "uci":
        normal = (df_test["holiday"] == 0) & (df_test["weathersit"] <= 2)
        holiday = (df_test["holiday"] == 1)
        extreme = (df_test["weathersit"] >= 3)
    else:
        holiday = (df_test["Holiday"].astype(str).str.strip() != "No Holiday")
        extreme = (df_test["Rainfall(mm)"] > 0) | (df_test["Snowfall (cm)"] > 0)
        normal = (~holiday) & (~extreme)

    out = {}
    for regime_name, mask in [("Normal", normal), ("Holiday", holiday), ("Extreme", extreme)]:
        m = mask.to_numpy()
        out[regime_name] = {
            "XGB_MAE": mae(y[m], xg_mid[m]),
            "XGB_P80": interval_coverage(y[m], xg_lo[m], xg_hi[m]),
            "TFT_MAE": mae(y[m], tft_mid[m]),
            "TFT_P80": interval_coverage(y[m], tft_lo[m], tft_hi[m]),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--download_data", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.download_data:
        paths = download_data(args.data_dir)
    else:
        paths = {
            "uci": os.path.join(args.data_dir, "hour.csv"),
            "seoul": os.path.join(args.data_dir, "SeoulBikeData.csv"),
        }

    # Prepare data
    uci = prepare_uci(paths["uci"])
    seoul = prepare_seoul(paths["seoul"])

    # Run experiments
    all_results = {}
    caches = {}
    for name, dataset in [("UCI", uci), ("Seoul", seoul)]:
        res, cache = run_dataset(name=name.lower(), data=dataset, out_dir=args.output_dir, seed=args.seed)
        all_results[name] = res
        caches[name] = cache

    # Regime table (XGB vs TFT only)
    all_regimes = {
        "UCI": regime_table("uci", uci, caches["UCI"]),
        "Seoul": regime_table("seoul", seoul, caches["Seoul"]),
    }

    payload = {"table2": all_results, "table3": all_regimes}
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Done. Results saved to", os.path.join(args.output_dir, "results.json"))

if __name__ == "__main__":
    main()
