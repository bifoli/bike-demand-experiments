from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UCI_HOUR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/hour.csv"
SEOUL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"

@dataclass
class PreparedData:
    # target in original count scale
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # standardized log1p target (used for model training)
    ylogz_train: np.ndarray
    ylogz_val: np.ndarray
    ylogz_test: np.ndarray

    # log1p train stats (for inverse transform)
    ylog_mu: float
    ylog_sigma: float

    # covariates per timestamp (standardized/one-hot)
    cov_train: np.ndarray
    cov_val: np.ndarray
    cov_test: np.ndarray

    # feature names for covariates
    cov_names: List[str]

    # datetimes for plotting / stratification
    dt_train: np.ndarray
    dt_val: np.ndarray
    dt_test: np.ndarray

    # original dataframe slices (optional)
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame

def download_file(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)

def download_data(data_dir: str) -> Dict[str, str]:
    """Download datasets into data_dir. Returns paths."""
    os.makedirs(data_dir, exist_ok=True)
    uci_path = os.path.join(data_dir, "hour.csv")
    seoul_path = os.path.join(data_dir, "SeoulBikeData.csv")
    if not os.path.exists(uci_path):
        download_file(UCI_HOUR_URL, uci_path)
    if not os.path.exists(seoul_path):
        download_file(SEOUL_URL, seoul_path)
    return {"uci": uci_path, "seoul": seoul_path}

def load_uci_hour(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # dteday is YYYY-MM-DD
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    # Ensure sorted
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def load_seoul(path: str) -> pd.DataFrame:
    # Seoul dataset is commonly encoded in latin-1 / cp1252
    df = pd.read_csv(path, encoding="latin1")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["datetime"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df = df.sort_values("datetime").reset_index(drop=True)
    # Keep only functioning days
    df = df[df["Functioning Day"].astype(str).str.strip().str.lower() == "yes"].reset_index(drop=True)
    return df

def _make_calendar_cols(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[datetime_col])
    out = df.copy()
    out["month"] = dt.dt.month.astype(int)
    out["weekday"] = dt.dt.weekday.astype(int)
    return out

def _prepare_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    continuous_cols: List[str],
    categorical_cols: List[str],
    binary_cols: List[str],
) -> PreparedData:
    # Calendar cols (month, weekday) may already be present; add if missing
    for split_df in (df_train, df_val, df_test):
        if "month" not in split_df.columns or "weekday" not in split_df.columns:
            pass

    # Targets
    y_train = df_train[target_col].to_numpy(dtype=float)
    y_val = df_val[target_col].to_numpy(dtype=float)
    y_test = df_test[target_col].to_numpy(dtype=float)

    # log1p + z-score using train stats
    ylog_train = np.log1p(y_train)
    ylog_val = np.log1p(y_val)
    ylog_test = np.log1p(y_test)
    mu = ylog_train.mean()
    sigma = ylog_train.std() + 1e-8
    ylogz_train = (ylog_train - mu) / sigma
    ylogz_val = (ylog_val - mu) / sigma
    ylogz_test = (ylog_test - mu) / sigma

    # Encoders
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    Xcat_train = ohe.fit_transform(df_train[categorical_cols].astype(str))
    Xcat_val = ohe.transform(df_val[categorical_cols].astype(str))
    Xcat_test = ohe.transform(df_test[categorical_cols].astype(str))

    Xcont_train = scaler.fit_transform(df_train[continuous_cols].astype(float))
    Xcont_val = scaler.transform(df_val[continuous_cols].astype(float))
    Xcont_test = scaler.transform(df_test[continuous_cols].astype(float))

    Xbin_train = df_train[binary_cols].to_numpy(dtype=float) if binary_cols else np.zeros((len(df_train),0))
    Xbin_val = df_val[binary_cols].to_numpy(dtype=float) if binary_cols else np.zeros((len(df_val),0))
    Xbin_test = df_test[binary_cols].to_numpy(dtype=float) if binary_cols else np.zeros((len(df_test),0))

    cov_train = np.concatenate([Xcat_train, Xcont_train, Xbin_train], axis=1)
    cov_val = np.concatenate([Xcat_val, Xcont_val, Xbin_val], axis=1)
    cov_test = np.concatenate([Xcat_test, Xcont_test, Xbin_test], axis=1)

    cov_names = list(ohe.get_feature_names_out(categorical_cols)) + continuous_cols + binary_cols

    return PreparedData(
        y_train=y_train, y_val=y_val, y_test=y_test,
        ylogz_train=ylogz_train, ylogz_val=ylogz_val, ylogz_test=ylogz_test,
        ylog_mu=float(mu), ylog_sigma=float(sigma),
        cov_train=cov_train, cov_val=cov_val, cov_test=cov_test,
        cov_names=cov_names,
        dt_train=df_train["datetime"].to_numpy(),
        dt_val=df_val["datetime"].to_numpy(),
        dt_test=df_test["datetime"].to_numpy(),
        df_train=df_train, df_val=df_val, df_test=df_test,
    )

def prepare_uci(path: str) -> PreparedData:
    df = load_uci_hour(path)
    df = _make_calendar_cols(df, "datetime")
    # Splits
    train = df[df["datetime"] < pd.Timestamp("2011-11-01")].copy()
    val = df[(df["datetime"] >= pd.Timestamp("2011-11-01")) & (df["datetime"] <= pd.Timestamp("2011-12-31 23:00:00"))].copy()
    test = df[(df["datetime"] >= pd.Timestamp("2012-01-01")) & (df["datetime"] <= pd.Timestamp("2012-12-31 23:00:00"))].copy()

    # Features (paper): season, holiday, workingday, weathersit, temp, atemp, hum, windspeed + hour/weekday/month
    categorical_cols = ["hr", "weekday", "month", "season", "weathersit"]
    continuous_cols = ["temp", "atemp", "hum", "windspeed"]
    binary_cols = ["holiday", "workingday"]

    return _prepare_features(
        train, val, test,
        target_col="cnt",
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

def prepare_seoul(path: str) -> PreparedData:
    df = load_seoul(path)
    df = _make_calendar_cols(df, "datetime")
    # Splits per manuscript
    train = df[df["datetime"] < pd.Timestamp("2018-06-01")].copy()
    val = df[(df["datetime"] >= pd.Timestamp("2018-06-01")) & (df["datetime"] < pd.Timestamp("2018-07-01"))].copy()
    test = df[(df["datetime"] >= pd.Timestamp("2018-07-01")) & (df["datetime"] < pd.Timestamp("2018-12-01"))].copy()

    # Continuous weather variables
    continuous_cols = [
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
        "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"
    ]
    categorical_cols = ["Hour", "weekday", "month", "Seasons", "Holiday"]
    binary_cols: List[str] = []  # Holiday is categorical in this dataset

    return _prepare_features(
        train, val, test,
        target_col="Rented Bike Count",
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        binary_cols=binary_cols,
    )

def make_tree_supervised(ylogz: np.ndarray, cov: np.ndarray, lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Lag features + covariates at prediction time."""
    X_list = []
    y_list = []
    for t in range(lookback, len(ylogz)):
        lags = ylogz[t - lookback:t]  # length lookback
        feat = np.concatenate([lags, cov[t]], axis=0)
        X_list.append(feat)
        y_list.append(ylogz[t])
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=float)

def make_seq_supervised(ylogz: np.ndarray, cov: np.ndarray, lookback: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Sequence input of length lookback+1 with masked demand at target step."""
    seq_len = lookback + 1
    X_list = []
    y_list = []
    for t in range(lookback, len(ylogz)):
        # past steps: (demand, obs_flag, cov...)
        past_y = ylogz[t - lookback:t]
        past_cov = cov[t - lookback:t]
        obs_flag = np.ones((lookback, 1), dtype=float)
        past = np.concatenate([past_y[:, None], obs_flag, past_cov], axis=1)

        # target step: demand masked, obs_flag=0, cov at t
        target_step = np.concatenate([np.zeros((1,1)), np.zeros((1,1)), cov[t:t+1]], axis=1)

        seq = np.concatenate([past, target_step], axis=0)
        assert seq.shape[0] == seq_len
        X_list.append(seq)
        y_list.append(ylogz[t])
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=float)
