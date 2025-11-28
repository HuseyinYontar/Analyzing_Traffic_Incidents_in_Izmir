# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# import warnings
# from path_getter import get_path_for_binned_directory_in  # <--- Added Import
#
# # Suppress warnings
# warnings.filterwarnings("ignore")
#
# # --- Load Excel File (Updated Routine) ---
# file_path = get_path_for_binned_directory_in()[3:]  # <--- Get path and slice
# df = pd.read_excel(file_path)  # <--- Load Excel
#
# # --- Preprocessing ---
# df.columns = df.columns.str.strip().str.upper()
# df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
#
# # Filter districts
# districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR",
#              "BUCA", "BALÇOVA", "ÇİĞLİ", "NARLIDERE", "KARŞIYAKA"]
# df = df[df["ILCE"].str.upper().isin(districts)]
#
# # --- Group by District and Month ---
# grouped = (
#     df.groupby(["ILCE", pd.Grouper(key="TARIH", freq="M")])
#     .agg(
#         UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
#         TOTAL_INCIDENTS=("CADDE", "size"),
#     )
# )
#
# # Calculate Ratio: Incidents per Distinct Street
# grouped["INCIDENTS_PER_STREET"] = grouped["TOTAL_INCIDENTS"] / grouped["UNIQUE_CADDE_COUNT"]
# grouped = grouped.reset_index()
#
# unique_districts = grouped["ILCE"].unique()
# n_districts = len(unique_districts)
#
# # --- Visualization Setup ---
# fig, axes = plt.subplots(n_districts, 1, figsize=(12, 4 * n_districts), sharex=False)
# if n_districts == 1: axes = [axes]
#
# # --- Analysis Loop ---
# for ax, district in zip(axes, unique_districts):
#     # Prepare Data
#     district_data = grouped[grouped["ILCE"] == district].set_index("TARIH")["INCIDENTS_PER_STREET"]
#
#     # Reindex for monthly frequency & Fill missing
#     full_idx = pd.date_range(start=district_data.index.min(), end=district_data.index.max(), freq='M')
#     district_data = district_data.reindex(full_idx).interpolate(method='linear')
#
#     # Split Data (Last 6 Months as Test)
#     if len(district_data) <= 6:
#         print(f"Skipping {district}: Not enough data.")
#         continue
#
#     train = district_data.iloc[:-6]
#     test = district_data.iloc[-6:]
#
#     # 1. Model: Exponential Smoothing
#     try:
#         # Use seasonality if history >= 2 years
#         if len(train) >= 24:
#             model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
#         else:
#             model = ExponentialSmoothing(train, trend="add").fit()
#         pred_es = model.forecast(len(test))
#     except:
#         # Fallback
#         model = ExponentialSmoothing(train).fit()
#         pred_es = model.forecast(len(test))
#
#     # 2. Baseline: Dummy Regressor (Historical Mean)
#     pred_dummy = pd.Series([train.mean()] * len(test), index=test.index)
#
#     # Calculate Errors
#     mae_es = mean_absolute_error(test, pred_es)
#     mae_dummy = mean_absolute_error(test, pred_dummy)
#
#     # --- Plotting ---
#     # Show last 2 years of training data + test
#     plot_start = train.index[-24] if len(train) > 24 else train.index[0]
#
#     ax.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='Train History', color='gray', alpha=0.6)
#     ax.plot(test.index, test, label='Actual (Test)', marker='o', color='black')
#     ax.plot(pred_es.index, pred_es, label=f'Exp. Smoothing (MAE={mae_es:.2f})', linestyle='--', color='red', marker='x')
#     ax.plot(pred_dummy.index, pred_dummy, label=f'Dummy Mean (MAE={mae_dummy:.2f})', linestyle=':', color='blue')
#
#     ax.set_title(f"{district}: Forecast Evaluation")
#     ax.set_ylabel("Incidents / Street")
#     ax.legend(loc="upper left")
#     ax.grid(True, linestyle='--', alpha=0.3)
#
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# import warnings
# import itertools
# from path_getter import get_path_for_binned_directory_in
#
# warnings.filterwarnings("ignore")
#
# # --- Load Data ---
# file_path = get_path_for_binned_directory_in()[3:]
# df = pd.read_excel(file_path)
#
# # --- Preprocessing ---
# df.columns = df.columns.str.strip().str.upper()
# df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
# districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR",
#              "BUCA", "BALÇOVA", "ÇİĞLİ", "NARLIDERE", "KARŞIYAKA"]
# df = df[df["ILCE"].str.upper().isin(districts)]
#
# grouped = (
#     df.groupby(["ILCE", pd.Grouper(key="TARIH", freq="M")])
#     .agg(UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
#          TOTAL_INCIDENTS=("CADDE", "size"))
# )
# grouped["INCIDENTS_PER_STREET"] = grouped["TOTAL_INCIDENTS"] / grouped["UNIQUE_CADDE_COUNT"]
# grouped = grouped.reset_index()
#
# unique_districts = grouped["ILCE"].unique()
# results = []
# n_districts = len(unique_districts)
# fig, axes = plt.subplots(n_districts, 1, figsize=(12, 4 * n_districts), sharex=False)
# if n_districts == 1: axes = [axes]
#
#
# # --- Optimization Function ---
# def optimize_sarima(series, seasonal=True):
#     # Simplified Grid Search
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq] if seasonal else [(0, 0, 0, 0)]
#
#     best_aic = float("inf")
#     best_model = None
#     best_params = None
#
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = SARIMAX(series, order=param, seasonal_order=param_seasonal,
#                               enforce_stationarity=False, enforce_invertibility=False)
#                 res = mod.fit(disp=False)
#                 if res.aic < best_aic:
#                     best_aic = res.aic
#                     best_model = res
#                     best_params = (param, param_seasonal)
#             except:
#                 continue
#     return best_model, best_params
#
#
# # --- Loop ---
# for ax, district in zip(axes, unique_districts):
#     data = grouped[grouped["ILCE"] == district].set_index("TARIH")["INCIDENTS_PER_STREET"]
#     full_idx = pd.date_range(start=data.index.min(), end=data.index.max(), freq='M')
#     data = data.reindex(full_idx).interpolate(method='linear')
#
#     if len(data) <= 6: continue
#     train, test = data.iloc[:-6], data.iloc[-6:]
#
#     # Check if enough data for seasonality
#     use_seasonal = len(train) >= 24
#     model, params = optimize_sarima(train, seasonal=use_seasonal)
#
#     if model:
#         pred = model.get_forecast(steps=len(test)).predicted_mean.clip(lower=0)
#         mae = mean_absolute_error(test, pred)
#         name = f"SARIMA{params[0]}x{params[1]}" if use_seasonal else f"ARIMA{params[0]}"
#     else:
#         pred = pd.Series([train.mean()] * len(test), index=test.index)
#         mae = mean_absolute_error(test, pred)
#         name = "Failed-Mean"
#
#     # Plot
#     ax.plot(train.index, train, label="Train")
#     ax.plot(test.index, test, label="Actual")
#     ax.plot(pred.index, pred, label=f"Forecast (MAE={mae:.2f})", linestyle="--")
#     ax.set_title(f"{district} - {name}")
#     ax.legend()
#
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error
# import numpy as np
# import warnings
# from path_getter import get_path_for_binned_directory_in
#
# warnings.filterwarnings("ignore")
#
# # --- Load Data ---
# file_path = get_path_for_binned_directory_in()[3:]
# df = pd.read_excel(file_path)
#
# # --- Preprocessing ---
# df.columns = df.columns.str.strip().str.upper()
# df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
# districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR",
#              "BUCA", "BALÇOVA", "ÇİĞLİ", "NARLIDERE", "KARŞIYAKA"]
# df = df[df["ILCE"].str.upper().isin(districts)]
#
# # Group & Calculate Ratio
# grouped = (
#     df.groupby(["ILCE", pd.Grouper(key="TARIH", freq="M")])
#     .agg(UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
#          TOTAL_INCIDENTS=("CADDE", "size"))
# )
# grouped["INCIDENTS_PER_STREET"] = grouped["TOTAL_INCIDENTS"] / grouped["UNIQUE_CADDE_COUNT"]
# grouped = grouped.reset_index()
#
# unique_districts = grouped["ILCE"].unique()
# results = []
# n_districts = len(unique_districts)
# fig, axes = plt.subplots(n_districts, 1, figsize=(12, 4 * n_districts), sharex=False)
# if n_districts == 1: axes = [axes]
#
# specified_order = (3, 1, 5)
#
# for ax, district in zip(axes, unique_districts):
#     # Prepare Data
#     data = grouped[grouped["ILCE"] == district].set_index("TARIH")["INCIDENTS_PER_STREET"]
#     full_idx = pd.date_range(start=data.index.min(), end=data.index.max(), freq='M')
#     data = data.reindex(full_idx).interpolate(method='linear')
#
#     if len(data) <= 6: continue
#     train, test = data.iloc[:-6], data.iloc[-6:]
#
#     # --- Fit ARIMA(3, 1, 5) ---
#     try:
#         model = SARIMAX(train,
#                         order=specified_order,
#                         seasonal_order=(0, 0, 0, 0),  # Pure ARIMA
#                         enforce_stationarity=False,
#                         enforce_invertibility=False)
#         model_fit = model.fit(disp=False, maxiter=200)
#
#         pred = model_fit.get_forecast(steps=len(test)).predicted_mean.clip(lower=0)
#         mae = mean_absolute_error(test, pred)
#         name = f"ARIMA{specified_order}"
#     except:
#         pred = pd.Series([train.mean()] * len(test), index=test.index)
#         mae = mean_absolute_error(test, pred)
#         name = "Failed"
#
#     # Baseline
#     mae_dummy = mean_absolute_error(test, pd.Series([train.mean()] * len(test)))
#
#     # Plot
#     ax.plot(train.index, train, label='Train')
#     ax.plot(test.index, test, label='Actual')
#     ax.plot(pred.index, pred, label=f'Forecast (MAE={mae:.2f})', linestyle='--')
#     ax.set_title(f"{district} - {name}")
#     ax.legend()
#
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
from path_getter import get_path_for_binned_directory_in

warnings.filterwarnings("ignore")

# --- Load Data ---
file_path = get_path_for_binned_directory_in()[3:]
df = pd.read_excel(file_path)

# --- Preprocessing ---
df.columns = df.columns.str.strip().str.upper()
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR",
             "BUCA", "BALÇOVA", "ÇİĞLİ", "NARLIDERE", "KARŞIYAKA"]
df = df[df["ILCE"].str.upper().isin(districts)]

grouped = (
    df.groupby(["ILCE", pd.Grouper(key="TARIH", freq="M")])
    .agg(UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
         TOTAL_INCIDENTS=("CADDE", "size"))
)
grouped["INCIDENTS_PER_STREET"] = grouped["TOTAL_INCIDENTS"] / grouped["UNIQUE_CADDE_COUNT"]
grouped = grouped.reset_index()

unique_districts = grouped["ILCE"].unique()
results = []
n_districts = len(unique_districts)
fig, axes = plt.subplots(n_districts, 1, figsize=(12, 4 * n_districts), sharex=False)
if n_districts == 1: axes = [axes]

# Combine User Params with Standard Seasonality
order = (3, 1, 5)
seasonal_order = (0, 1, 1, 12)

for ax, district in zip(axes, unique_districts):
    data = grouped[grouped["ILCE"] == district].set_index("TARIH")["INCIDENTS_PER_STREET"]
    full_idx = pd.date_range(start=data.index.min(), end=data.index.max(), freq='M')
    data = data.reindex(full_idx).interpolate(method='linear')

    if len(data) <= 6: continue
    train, test = data.iloc[:-6], data.iloc[-6:]

    try:
        if len(train) >= 24:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, maxiter=200)
            model_name = "SARIMA(0,1,1,12)"
        else:
            # Fallback for short history
            model = SARIMAX(train, order=order, seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False, maxiter=200)
            model_name = "ARIMA(No Seas)"

        pred = model_fit.get_forecast(steps=len(test)).predicted_mean.clip(lower=0)
    except:
        pred = pd.Series([train.mean()] * len(test), index=test.index)
        model_name = "Failed"

    mae = mean_absolute_error(test, pred)

    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Actual')
    ax.plot(pred.index, pred, label=f'Forecast (MAE={mae:.2f})', linestyle='--')
    ax.set_title(f"{district} - {model_name}")
    ax.legend()

plt.tight_layout()
plt.show()