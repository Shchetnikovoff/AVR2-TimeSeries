"""Пайплайн прогноза суточного минимума температуры (Задача 4).
Загрузка -> предобработка -> обучение -> прогноз h=90 -> оценка + замер времени.
Лучшая детерминированная конфигурация из ноутбука: HoltWinters(аддитивный, m=365).
Запуск: .venv/bin/python pipeline.py
"""
import time

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import HoltWinters, SeasonalNaive

H = 90


def load(path="data/melbourne_tmin_daily_processed.csv"):
    df = pd.read_csv(path, parse_dates=["ds"])
    return df


def preprocess(df):
    df = df.sort_values("ds").reset_index(drop=True)
    full = pd.DataFrame({"ds": pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")})
    df = full.merge(df, on="ds", how="left")
    df["y"] = df["y"].interpolate(method="linear")
    df["unique_id"] = "melb"
    return df


def train_predict(train, h=H):
    sf = StatsForecast(models=[HoltWinters(season_length=365),
                               SeasonalNaive(season_length=365)], freq="D")
    return sf.forecast(df=train, h=h)


def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    m = np.abs(y_true) >= 0.5
    mape = float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)
    return rmse, mae, mape


def main():
    t0 = time.time()
    df = preprocess(load())
    train, test = df.iloc[:-H], df.iloc[-H:]
    t1 = time.time()
    fc = train_predict(train)
    t2 = time.time()
    for model in ["HoltWinters", "SeasonalNaive"]:
        rmse, mae, mape = evaluate(test["y"].values, fc[model].values)
        print(f"{model}: RMSE={rmse:.3f} MAE={mae:.3f} MAPE={mape:.2f}%")
    print(f"время: загрузка+предобработка {t1 - t0:.1f} c, "
          f"обучение+прогноз {t2 - t1:.1f} c, всего {t2 - t0:.1f} c")


if __name__ == "__main__":
    main()
