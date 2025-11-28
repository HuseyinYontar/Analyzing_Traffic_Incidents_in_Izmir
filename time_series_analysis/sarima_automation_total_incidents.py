import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
import itertools
import os

# --- 1. CONFIGURATION & IMPORTS ---
warnings.filterwarnings("ignore")

# Target Districts for specific analysis
TARGET_DISTRICTS = [
    "ÇİĞLİ", "KARŞIYAKA", "BORNOVA", "BALÇOVA", "NARLIDERE",
    "KONAK", "BUCA", "BAYRAKLI", "KARABAĞLAR", "GAZİEMİR"
]

# Frequency Configurations
FREQUENCIES = {
    'Daily': {'freq': 'D', 'period': 7, 'test_size': 30},
    'Weekly': {'freq': 'W-MON', 'period': 52, 'test_size': 12},
    'Monthly': {'freq': 'M', 'period': 12, 'test_size': 6}
}


# --- 2. DATA LOADING & PREPROCESSING ---
def normalize_turkish_chars(text):
    if not isinstance(text, str): return text
    replacements = {
        'i': 'İ', 'ı': 'I', 'ğ': 'Ğ', 'ü': 'Ü', 'ş': 'Ş', 'ö': 'Ö', 'ç': 'Ç',
        'I': 'I', 'İ': 'İ'
    }
    result = ""
    for char in text:
        result += replacements.get(char, char.upper())
    return result


def load_data():
    try:
        from path_getter import get_path_for_binned_directory_in
        file_path = get_path_for_binned_directory_in()[3:]
        print(f"Loading data from: {file_path}")
        df = pd.read_excel(file_path)
    except ImportError:
        print("Warning: 'path_getter' not found. Trying local CSV...")
        file_path = "izbb-kaza-ariza-verileri-SON-binned-frequent-accident_types.xlsx - Sheet1.csv"
        try:
            df = pd.read_csv(file_path)
        except:
            print("Error: Could not find data file.")
            return None

    df.columns = df.columns.str.strip().str.upper()
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

    # Normalize ILCE for filtering
    df["ILCE_NORM"] = df["ILCE"].astype(str).apply(normalize_turkish_chars)

    return df


# --- 3. METRICS & MODEL FUNCTIONS ---

def calculate_safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def fit_sarimax_grid(train, p_rng, d_rng, q_rng, seasonal_order_list):
    best_aic = float("inf")
    best_model = None
    best_params = ""

    # Limit grid size for speed if needed, but keeping it robust here
    pdq = list(itertools.product(p_rng, d_rng, q_rng))

    for param in pdq:
        for s_param in seasonal_order_list:
            try:
                mod = SARIMAX(train, order=param, seasonal_order=s_param,
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_model = res
                    best_params = f"Order={param}, Seas={s_param}"
            except:
                continue
    return best_model, best_params


def get_exponential_smoothing(train, period):
    configs = [
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': period},
        {'trend': 'add', 'seasonal': None},
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': period}
    ]
    best_aic = float("inf")
    best_model = None
    best_params = ""

    for cfg in configs:
        if cfg['seasonal'] is not None and (period < 2 or len(train) < 2 * period): continue
        try:
            model = ExponentialSmoothing(train, trend=cfg['trend'], seasonal=cfg['seasonal'],
                                         seasonal_periods=cfg.get('seasonal_periods')).fit()
            score = model.aic if hasattr(model, 'aic') else model.sse
            if score < best_aic:
                best_aic = score
                best_model = model
                best_params = str(cfg)
        except:
            continue
    return best_model, best_params


# --- 4. CORE ANALYSIS ENGINE ---

def find_best_model_for_series(name, series, freq_name, freq_config):
    """
    Runs the grid search on a single time series and returns the best model results.
    """
    test_size = freq_config['test_size']
    period = freq_config['period']

    if len(series) < (test_size * 2):
        print(f"  [Skipping {name}] Not enough data (len={len(series)}).")
        return None

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    # Define Search Grid
    models_to_run = []

    # 1. SARIMA Families
    # Standard ARIMA/SARIMA grids
    models_to_run.append(("ARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), [(0, 0, 0, 0)])))

    if len(train) >= (2 * period) and period > 1:
        s_grid = [(1, 0, 0, period), (0, 1, 0, period), (1, 1, 0, period)]
        models_to_run.append(("SARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), s_grid)))

    # 2. Exponential Smoothing (for comparison)
    models_to_run.append(("ExpSmoothing", get_exponential_smoothing, (train, period)))

    # Baseline
    baseline_pred = pd.Series([train.mean()] * test_size, index=test.index)
    baseline_mae = mean_absolute_error(test, baseline_pred)
    baseline_mape = calculate_safe_mape(test, baseline_pred)

    best_result = {
        "Name": name, "Frequency": freq_name, "Model_Type": "Baseline",
        "Params": "Mean", "MAE": baseline_mae, "MAPE": baseline_mape,
        "Preds": baseline_pred, "Test": test, "Train": train, "Improvement_Pct": 0.0
    }

    print(f"  > Analyzing {name} ({freq_name})... ", end="")

    for m_name, m_func, m_args in models_to_run:
        try:
            model_obj, params_str = m_func(*m_args)
            if model_obj:
                pred = model_obj.forecast(test_size)
                if hasattr(pred, "predicted_mean"): pred = pred.predicted_mean
                pred = pd.Series(pred.values, index=test.index).clip(lower=0)

                mae = mean_absolute_error(test, pred)
                mape = calculate_safe_mape(test, pred)

                if mae < best_result["MAE"]:
                    best_result.update({
                        "Model_Type": m_name, "Params": params_str,
                        "MAE": mae, "MAPE": mape, "Preds": pred,
                        "Improvement_Pct": ((baseline_mae - mae) / baseline_mae) * 100
                    })
        except Exception:
            continue

    print(f"Winner: {best_result['Model_Type']} (MAE: {best_result['MAE']:.2f})")
    return best_result


def plot_best_result(result, output_folder="analysis_plots"):
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    train, test, preds = result["Train"], result["Test"], result["Preds"]
    name, freq = result["Name"], result["Frequency"]

    plt.figure(figsize=(12, 6))

    # Smart Lookback
    lookback = max(len(test) * 4, 30)
    plot_start = train.index[-lookback] if len(train) > lookback else train.index[0]

    plt.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='History', color='gray', alpha=0.5)
    plt.plot(test.index, test, label='Actual', marker='o', color='black')
    plt.plot(preds.index, preds,
             label=f'WINNER: {result["Model_Type"]}\nMAE={result["MAE"]:.2f} | MAPE={result["MAPE"]:.1f}%',
             color='green', linestyle='--', marker='x')

    plt.title(f"{name} ({freq}) - Best Forecast Model")
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_name = name.replace(" ", "_").replace("İ", "I")
    plt.savefig(f"{output_folder}/{safe_name}_{freq}_Best.png")
    plt.close()


# --- 5. MAIN ROUTINE ---
def run_analysis():
    df = load_data()
    if df is None: return

    summary_rows = []

    # --- PHASE 1: TOTAL IZMIR ANALYSIS ---
    print("\n" + "=" * 40)
    print("PHASE 1: TOTAL IZMIR ANALYSIS (ALL DISTRICTS)")
    print("=" * 40)

    for freq_name, freq_config in FREQUENCIES.items():
        # 1. Resample entire dataset
        total_ts = df.set_index("TARIH").resample(freq_config['freq']).size()

        # 2. Fill Missing
        if freq_name == 'Daily':
            ts_nan = total_ts.replace(0, np.nan)
            weekly_means = ts_nan.groupby(pd.Grouper(freq='W')).transform('mean')
            total_ts = ts_nan.fillna(weekly_means).fillna(0)
        else:
            total_ts = total_ts.fillna(0)

        # 3. Analyze
        res = find_best_model_for_series("TOTAL_IZMIR", total_ts, freq_name, freq_config)
        if res:
            summary_rows.append(res)
            plot_best_result(res, "total_izmir_plots")

    # --- PHASE 2: DISTRICT ANALYSIS ---
    print("\n" + "=" * 40)
    print("PHASE 2: TARGET DISTRICT ANALYSIS")
    print("=" * 40)

    districts_norm = [normalize_turkish_chars(d) for d in TARGET_DISTRICTS]

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n--- Frequency: {freq_name} ---")

        for district, dist_norm in zip(TARGET_DISTRICTS, districts_norm):
            # 1. Filter & Resample
            dist_data = df[df["ILCE_NORM"] == dist_norm]
            dist_ts = dist_data.set_index("TARIH").resample(freq_config['freq']).size()

            # 2. Fill Missing
            if freq_name == 'Daily':
                ts_nan = dist_ts.replace(0, np.nan)
                weekly_means = ts_nan.groupby(pd.Grouper(freq='W')).transform('mean')
                dist_ts = ts_nan.fillna(weekly_means).fillna(0)
            else:
                dist_ts = dist_ts.fillna(0)

            # 3. Analyze
            res = find_best_model_for_series(district, dist_ts, freq_name, freq_config)
            if res:
                summary_rows.append(res)
                plot_best_result(res, "district_analysis_plots")

    # --- SAVE SUMMARY ---
    # Convert list of dicts to DF, drop the heavy series objects for CSV
    export_data = [{k: v for k, v in r.items() if k not in ['Preds', 'Test', 'Train']} for r in summary_rows]
    res_df = pd.DataFrame(export_data)

    # Reorder columns
    cols = ['Name', 'Frequency', 'Model_Type', 'Params', 'MAE', 'MAPE', 'Improvement_Pct']
    res_df = res_df[cols]

    res_df.to_csv("analysis_summary_complete.csv", index=False)

    print("\nAnalysis Complete. Results saved to 'analysis_summary_complete.csv'.")
    print("Plots saved in 'total_izmir_plots' and 'district_analysis_plots'.")


if __name__ == "__main__":
    run_analysis()