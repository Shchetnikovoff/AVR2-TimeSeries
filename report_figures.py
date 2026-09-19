"""
report_figures.py — дополнение к ноутбуку для верификации и медиаматериалов отчёта.

Воспроизводит все 14 конфигураций моделей из AVR2_Time_Series_Analysis.ipynb
(те же параметры, seed 42) и строит:
  img/09_forecasts_all.png   — прогноз каждой из 14 моделей на holdout против факта
  img/10_backtest_windows.png — RMSE по каждому из 3 окон бектеста для всех моделей
  img/11_dm_heatmap.png      — матрица p-value теста Диболда-Мариано (holdout, HAC)
Сохраняет:
  results/backtest_windows.csv — RMSE по окнам
  results/dm_test.csv          — p-value DM-теста против лучшей модели
  results/model_configs.json   — полные конфигурации, версии библиотек, seed

Запуск: python report_figures.py  (из корня репозитория)
"""
import json
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import torch

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
np.random.seed(42)
torch.manual_seed(42)

from statsforecast import StatsForecast
from statsforecast.models import (ARIMA, AutoARIMA, AutoETS, AutoTheta,
                                  HoltWinters, Naive, SeasonalNaive, Theta)
from statsforecast.utils import AirPassengers as _ap  # noqa: F401 (импорт для проверки версии)

H = 90
SEED = 42

df = pd.read_csv("data/melbourne_tmin_daily_processed.csv", parse_dates=["ds"])
df["unique_id"] = "melb"
train = df.iloc[:-H].copy()
test = df.iloc[-H:].copy()
y_true = test["y"].values


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------- статметоды: те же конфиги, что в ноутбуке ----------------
stat_models = [Naive(),
               SeasonalNaive(season_length=365),
               ARIMA(order=(7, 0, 0)),
               AutoARIMA(season_length=7),
               HoltWinters(season_length=365),
               AutoETS(season_length=365, model="ZZZ"),
               Theta(season_length=365),
               AutoTheta(season_length=365)]
stat_names = ["Naive", "SeasonalNaive", "ARIMA", "AutoARIMA",
              "HoltWinters", "AutoETS", "Theta", "AutoTheta"]
t0 = time.time()
sf = StatsForecast(models=stat_models, freq="D")
fc_stat = sf.forecast(df=train, h=H, level=[80, 95])
t_stat = time.time() - t0

# ---------------- ML: те же конфиги ----------------
from mlforecast import MLForecast
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge

mlf = MLForecast(
    models={"RF": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            "HGB": HistGradientBoostingRegressor(random_state=SEED),
            "Ridge": Ridge()},
    freq="D",
    lags=[1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 90, 180, 364, 365],
    date_features=["dayofweek", "month", "quarter", "dayofyear"],
    num_threads=1,
)
t0 = time.time()
mlf.fit(train)
fc_ml = mlf.predict(H)
t_ml = time.time() - t0

# ---------------- DL: те же конфиги ----------------
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, MLP, NHITS

dl_models = [LSTM(h=H, input_size=365, max_steps=500, random_seed=SEED),
             MLP(h=H, input_size=365, max_steps=500, random_seed=SEED),
             NHITS(h=H, input_size=365, max_steps=500, random_seed=SEED)]
nf = NeuralForecast(models=dl_models, freq="D")
t0 = time.time()
nf.fit(df=train)
fc_dl = nf.predict()
t_dl = time.time() - t0

# ---------------- свод прогнозов ----------------
forecasts = {n: fc_stat[n].values for n in stat_names}
forecasts.update({"ML-" + m: fc_ml[m].values for m in ["RF", "HGB", "Ridge"]})
forecasts.update({"DL-" + m: fc_dl[m].values for m in ["LSTM", "MLP", "NHITS"]})
holdout = pd.read_csv("results/metrics_holdout.csv")
order = holdout["model"].tolist()

# ---------------- рис. 9: все 14 прогнозов ----------------
fig, axes = plt.subplots(4, 4, figsize=(16, 13), sharex=True, sharey=True)
axes = axes.ravel()
for i, name in enumerate(order):
    ax = axes[i]
    ax.plot(test["ds"], y_true, "k-", lw=1.3, label="факт")
    ax.plot(test["ds"], forecasts[name], lw=1.3, color="tab:blue", label="модель")
    met = holdout[holdout["model"] == name].iloc[0]
    ax.set_title(f"{name}\nRMSE={met['RMSE']:.2f} MAE={met['MAE']:.2f}", fontsize=10)
    if i == 0:
        ax.legend(fontsize=8)
for j in range(len(order), 16):
    axes[j].axis("off")
fig.suptitle("Прогнозы всех 14 моделей на holdout (90 дней) против факта", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig("img/09_forecasts_all.png")
plt.close(fig)
print("img/09_forecasts_all.png")

# ---------------- бектест по окнам ----------------
cv_stat = sf.cross_validation(df=df, h=H, n_windows=3, step_size=H)
cv_ml = mlf.cross_validation(df=df, h=H, n_windows=3, step_size=H)
cv_dl = nf.cross_validation(df=df, n_windows=3, step_size=H, h=H)


def per_window(cv, names, prefix=""):
    y_idx = df.set_index("ds")["y"]
    rows = []
    for m in names:
        for cutoff, w in cv.groupby("cutoff"):
            yy = y_idx.loc[w["ds"]].values
            rows.append({"model": prefix + m, "cutoff": str(pd.Timestamp(cutoff).date()),
                         "RMSE": rmse(yy, w[m].values)})
    return rows


rows = per_window(cv_stat, stat_names)
rows += per_window(cv_ml, ["RF", "HGB", "Ridge"], "ML-")
rows += per_window(cv_dl, ["LSTM", "MLP", "NHITS"], "DL-")
bw = pd.DataFrame(rows)
bw.to_csv("results/backtest_windows.csv", index=False)

piv = bw.pivot(index="cutoff", columns="model", values="RMSE")[order]
fig, ax = plt.subplots(figsize=(13, 6))
markers = ["o", "s", "^"]
for k, name in enumerate(order):
    ax.plot(piv.index, piv[name].values, marker=markers[k % 3], lw=1.2, label=name)
ax.set_title("RMSE по окнам бектеста (3 окна по 90 дней, сдвиг 90)")
ax.set_xlabel("окно (cutoff)")
ax.set_ylabel("RMSE, C")
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig("img/10_backtest_windows.png")
plt.close(fig)
print("img/10_backtest_windows.png")

# ---------------- тест Диболда-Мариано (HAC, лаг 3) ----------------
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: F401


def dm_pvalue(e1, e2, h=3):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    def gamma(k):
        return np.mean((d[: n - k] - dbar) * (d[k:] - dbar))
    var = gamma(0)
    for k in range(1, h):
        var += 2.0 * gamma(k)
    if var <= 0:
        return 1.0
    stat = dbar / np.sqrt(var / n)
    from scipy import stats as st
    return 2 * (1 - st.norm.cdf(abs(stat)))


best_name = order[0]
e_best = y_true - forecasts[best_name]
dm_rows = []
for name in order[1:]:
    e = y_true - forecasts[name]
    dm_rows.append({"vs_best": best_name, "model": name,
                    "dm_pvalue": dm_pvalue(e, e_best)})
dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv("results/dm_test.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
lbl = [r["model"] for r in dm_rows]
vals = [r["dm_pvalue"] for r in dm_rows]
colors = ["tab:red" if v < 0.05 else "tab:gray" for v in vals]
ax.barh(lbl, vals, color=colors)
ax.axvline(0.05, color="k", ls="--", lw=1)
ax.set_xlabel("p-value теста Диболда-Мариано против лучшей модели (HAC, лаг 3)")
ax.set_title(f"Значимость отличий от {best_name} на holdout (красное: значимо, p<0.05)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("img/11_dm_heatmap.png")
plt.close(fig)
print("img/11_dm_heatmap.png")

# ---------------- полный дамп конфигураций ----------------
import scipy
import neuralforecast
import mlforecast
import statsforecast

configs = {
    "seed": SEED,
    "hardware": platform.machine() + ", " + platform.platform(),
    "python": sys.version.split()[0],
    "versions": {
        "pandas": pd.__version__, "numpy": np.__version__,
        "statsforecast": statsforecast.__version__,
        "mlforecast": mlforecast.__version__,
        "neuralforecast": neuralforecast.__version__,
        "neuralforecast_torch": torch.__version__,
        "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
    },
    "split": {"train_days": int(len(train)), "holdout_days": H,
              "train_end": str(train["ds"].max().date()),
              "holdout_start": str(test["ds"].min().date())},
    "backtest": {"n_windows": 3, "h": H, "step_size": H},
    "models": {
        "Naive": {"type": "baseline", "params": {"": "последнее значение"}},
        "SeasonalNaive": {"type": "baseline", "params": {"season_length": 365}},
        "ARIMA": {"type": "stat, ручной", "params": {"order": [7, 0, 0]},
                  "обоснование": "d=0 по ADF p=0.00025, AR(7) по PACF"},
        "AutoARIMA": {"type": "stat, авто", "params": {"season_length": 7}},
        "HoltWinters": {"type": "stat, ручной", "params": {"season_length": 365,
                                                          "error_type": "additive (по умолчанию)"}},
        "AutoETS": {"type": "stat, авто", "params": {"season_length": 365, "model": "ZZZ"}},
        "Theta": {"type": "stat, ручной", "params": {"season_length": 365}},
        "AutoTheta": {"type": "stat, авто", "params": {"season_length": 365}},
        "ML-RF": {"type": "ML", "params": {"n_estimators": 200, "random_state": SEED, "n_jobs": -1}},
        "ML-HGB": {"type": "ML", "params": {"HistGradientBoostingRegressor": "default", "random_state": SEED}},
        "ML-Ridge": {"type": "ML", "params": {"alpha": "1.0 (default)"}},
        "DL-LSTM": {"type": "DL", "params": {"h": H, "input_size": 365, "max_steps": 500,
                                             "random_seed": SEED, "остальное": "default"}},
        "DL-MLP": {"type": "DL", "params": {"h": H, "input_size": 365, "max_steps": 500,
                                            "random_seed": SEED, "остальное": "default"}},
        "DL-NHITS": {"type": "DL", "params": {"h": H, "input_size": 365, "max_steps": 500,
                                              "random_seed": SEED, "остальное": "default"}},
    },
    "mlforecast_features": {
        "lags": [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 90, 180, 364, 365],
        "date_features": ["dayofweek", "month", "quarter", "dayofyear"],
    },
    "runtimes_sec": {"stat_holdout": round(t_stat, 1), "ml_holdout": round(t_ml, 1),
                     "dl_holdout": round(t_dl, 1)},
}
with open("results/model_configs.json", "w", encoding="utf-8") as f:
    json.dump(configs, f, ensure_ascii=False, indent=2)
print("results/model_configs.json")
print("Готово. DM против", best_name, ":")
print(dm_df.round(4).to_string(index=False))
