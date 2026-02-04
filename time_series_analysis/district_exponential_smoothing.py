import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import os

# CONFIGURATION & IMPORTS
warnings.filterwarnings("ignore")

# Target Districts
TARGET_DISTRICTS = [
    "ÇİĞLİ",
    "KARŞIYAKA",
    "BORNOVA",
    "BALÇOVA",
    "NARLIDERE",
    "KONAK",
    "BUCA",
    "BAYRAKLI",
    "KARABAĞLAR",
    "GAZİEMİR"
]

# Frequency Configurations
FREQUENCIES = {
    'Daily': {'freq': 'D', 'period': 7, 'test_size': 30},
    'Weekly': {'freq': 'W-MON', 'period': 52, 'test_size': 12},
    'Monthly': {'freq': 'M', 'period': 12, 'test_size': 6}
}


# DATA LOADING & PREPROCESSING
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
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    df.columns = df.columns.str.strip().str.upper()
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

    df["ILCE_NORM"] = df["ILCE"].astype(str).apply(normalize_turkish_chars)
    targets_norm = [normalize_turkish_chars(d) for d in TARGET_DISTRICTS]
    df = df[df["ILCE_NORM"].isin(targets_norm)]

    return df


# METRIC FUNCTIONS
def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles division by zero by filtering out zero values in y_true.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Mask zeros to avoid infinity
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan  # Undefined if all actuals are 0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# EXPONENTIAL SMOOTHING AUTOMATION
def find_best_es_model(train, period):
    """
    Grid searches for the best Exponential Smoothing configuration (Trend/Seasonal).
    Selection based on AIC (or SSE if AIC unavailable).
    """
    # Define configurations to test
    # Note: We avoid 'mul' (multiplicative) because data likely contains 0s
    configs = [
        {'trend': None, 'seasonal': None, 'name': 'Simple Exp Smoothing'},
        {'trend': 'add', 'seasonal': None, 'name': 'Holt’s Linear (Trend)'},
        {'trend': None, 'seasonal': 'add', 'name': 'Seasonal Only'},
        {'trend': 'add', 'seasonal': 'add', 'name': 'Holt-Winters (Trend+Seas)'}
    ]

    best_score = float("inf")
    best_model = None
    best_cfg_name = "None"

    for cfg in configs:
        # Skip seasonal configs if period is invalid or data too short
        if cfg['seasonal'] is not None and (period < 2 or len(train) < 2 * period):
            continue

        try:
            model = ExponentialSmoothing(train,
                                         trend=cfg['trend'],
                                         seasonal=cfg['seasonal'],
                                         seasonal_periods=period).fit()

            # Use AIC if available, else SSE
            score = model.aic if hasattr(model, 'aic') else model.sse

            if score < best_score:
                best_score = score
                best_model = model
                best_cfg_name = cfg['name']
        except:
            continue

    return best_model, best_cfg_name


# MAIN ROUTINE
def run_es_analysis():
    df = load_data()
    if df is None: return

    results_summary = []

    if not os.path.exists("es_analysis_plots"):
        os.makedirs("es_analysis_plots")

    print(f"\nStarting Exponential Smoothing Analysis (with MAPE)...")

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n=== Processing {freq_name} Data ===")

        for district in TARGET_DISTRICTS:
            dist_norm = normalize_turkish_chars(district)
            print(f"  Analyzing {district}...")

            # --- DATA PREP ---
            dist_data = df[df["ILCE_NORM"] == dist_norm]
            ts_resampled = dist_data.set_index("TARIH").resample(freq_config['freq']).size()

            if freq_name == 'Daily':
                ts_nan = ts_resampled.replace(0, np.nan)
                weekly_means = ts_nan.groupby(pd.Grouper(freq='W')).transform('mean')
                ts = ts_nan.fillna(weekly_means).fillna(0)
            else:
                ts = ts_resampled.fillna(0)

            test_size = freq_config['test_size']
            if len(ts) < (test_size * 2):
                print(f"    Skipping: Not enough data.")
                continue

            train = ts.iloc[:-test_size]
            test = ts.iloc[-test_size:]
            period = freq_config['period']

            # --- FIND BEST ES MODEL ---
            best_model, model_name = find_best_es_model(train, period)

            if best_model:
                pred = best_model.forecast(test_size)
                pred = pd.Series(pred.values, index=test.index).clip(lower=0)
            else:
                pred = pd.Series([train.mean()] * test_size, index=test.index)
                model_name = "Failed (Mean)"

            # --- BASELINE ---
            baseline_pred = pd.Series([train.mean()] * test_size, index=test.index)

            # --- CALCULATE METRICS (MAE, RMSE, MAPE) ---
            # Model Metrics
            mae = mean_absolute_error(test, pred)
            rmse = np.sqrt(mean_squared_error(test, pred))
            mape = calculate_mape(test, pred)

            # Baseline Metrics
            b_mae = mean_absolute_error(test, baseline_pred)
            b_rmse = np.sqrt(mean_squared_error(test, baseline_pred))
            b_mape = calculate_mape(test, baseline_pred)

            imp_pct = ((b_mae - mae) / b_mae) * 100

            results_summary.append({
                "District": district,
                "Frequency": freq_name,
                "Best_Model": model_name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE_Pct": round(mape, 2) if not np.isnan(mape) else "NaN",
                "Baseline_MAE": round(b_mae, 2),
                "Baseline_MAPE": round(b_mape, 2) if not np.isnan(b_mape) else "NaN",
                "Improvement_MAE_Pct": round(imp_pct, 2)
            })

            # --- PLOTTING ---
            plt.figure(figsize=(12, 6))

            # Safe lookback for plotting
            lookback = max(test_size * 4, 20)
            if len(train) > lookback:
                plot_start = train.index[-lookback]
            else:
                plot_start = train.index[0]

            plt.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='History', color='gray', alpha=0.5)
            plt.plot(test.index, test, label='Actual', marker='o', color='black')

            label_text = f"Best ES: {model_name}\nMAE={mae:.1f}, MAPE={mape:.1f}%"
            plt.plot(pred.index, pred, label=label_text, color='purple', linestyle='--', marker='x', linewidth=2)

            plt.plot(baseline_pred.index, baseline_pred, label=f'Baseline (Mean)', color='blue', linestyle=':',
                     alpha=0.6)

            plt.title(f"{district} ({freq_name}): Exponential Smoothing Forecast")
            plt.ylabel("Incident Volume")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.3)

            safe_name = dist_norm.replace(" ", "_").replace("İ", "I").replace("Ş", "S").replace("Ç", "C").replace("Ğ",
                                                                                                                  "G").replace(
                "Ö", "O").replace("Ü", "U")
            plt.savefig(f"es_analysis_plots/{safe_name}_{freq_name}_ES.png")
            plt.close()

    # --- SAVE RESULTS ---
    res_df = pd.DataFrame(results_summary)
    res_df.to_csv("district_es_results_mape.csv", index=False)

    print("\n" + "=" * 50)
    print("EXPONENTIAL SMOOTHING ANALYSIS COMPLETE")
    print("=" * 50)
    print("Top Models by MAE Improvement:")
    print(res_df.sort_values("Improvement_MAE_Pct", ascending=False).head(10)[
              ['District', 'Frequency', 'Best_Model', 'MAE', 'MAPE_Pct']])
    print("\nResults saved to 'district_es_results_mape.csv'")


if __name__ == "__main__":
    run_es_analysis()