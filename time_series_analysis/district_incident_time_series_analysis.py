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


# --- 2. DATA LOADING & PREPROCESSING ---
def normalize_turkish_chars(text):
    """
    Handles reliable uppercase conversion for Turkish characters.
    """
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

    print(f"Found data for districts: {df['ILCE_NORM'].unique()}")
    return df


# --- 3. METRICS & MODEL FUNCTIONS ---

def calculate_safe_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles division by zero by ignoring time steps where y_true is 0.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Create mask where actual value is NOT zero
    mask = y_true != 0

    # If all actuals are zero, MAPE is undefined (return NaN or 0)
    if np.sum(mask) == 0:
        return np.nan

    # Calculate MAPE only on non-zero indices
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def fit_sarimax_grid(train, p_rng, d_rng, q_rng, seasonal_order_list):
    """Generic grid search for AR, MA, ARMA, ARIMA, SARIMA."""
    best_aic = float("inf")
    best_model = None
    best_params = ""

    pdq = list(itertools.product(p_rng, d_rng, q_rng))

    for param in pdq:
        for s_param in seasonal_order_list:
            try:
                mod = SARIMAX(train,
                              order=param,
                              seasonal_order=s_param,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_model = res
                    best_params = f"Order={param}, Seas={s_param}"
            except:
                continue
    return best_model, best_params


def get_exponential_smoothing(train, period):
    """Fits Exponential Smoothing (Holt-Winters)."""
    configs = [
        {'trend': 'add', 'seasonal': None},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': period},
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': period},
        {'trend': None, 'seasonal': None}
    ]

    best_aic = float("inf")
    best_model = None
    best_params = ""

    for cfg in configs:
        if cfg['seasonal'] is not None and (period < 2 or len(train) < 2 * period):
            continue

        try:
            model = ExponentialSmoothing(train,
                                         trend=cfg['trend'],
                                         seasonal=cfg['seasonal'],
                                         seasonal_periods=cfg.get('seasonal_periods')).fit()
            score = model.aic if hasattr(model, 'aic') else model.sse
            if score < best_aic:
                best_aic = score
                best_model = model
                best_params = str(cfg)
        except:
            continue

    return best_model, best_params


# --- 4. MAIN ANALYSIS ROUTINE ---
def run_analysis():
    df = load_data()
    if df is None: return

    results_summary = []

    if not os.path.exists("district_analysis_plots"):
        os.makedirs("district_analysis_plots")

    print(f"\nStarting Analysis for {len(TARGET_DISTRICTS)} districts...")

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n=== Processing {freq_name} Data ===")

        for district in TARGET_DISTRICTS:
            dist_norm = normalize_turkish_chars(district)
            print(f"  Analyzing {district}...")

            # --- PREPARE DATA ---
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

            models_to_run = []

            # (Model Definitions kept same as before)
            models_to_run.append(("AR", fit_sarimax_grid, (train, range(1, 3), [0], [0], [(0, 0, 0, 0)])))
            models_to_run.append(("MA", fit_sarimax_grid, (train, [0], [0], range(1, 3), [(0, 0, 0, 0)])))
            models_to_run.append(("ARMA", fit_sarimax_grid, (train, range(1, 2), [0], range(1, 2), [(0, 0, 0, 0)])))
            models_to_run.append(
                ("ARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), [(0, 0, 0, 0)])))

            use_seasonal = len(train) >= (2 * period)
            s_grid = [(1, 0, 0, period), (0, 1, 0, period), (0, 0, 0, 0)] if use_seasonal and period > 1 else [
                (0, 0, 0, 0)]
            models_to_run.append(("SARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), s_grid)))

            models_to_run.append(("ExpSmoothing", get_exponential_smoothing, (train, period)))

            # --- RUN & COMPARE ---
            best_mae = float("inf")
            winner_name = "None"
            winner_preds = None

            # Baseline Metrics
            baseline_pred = pd.Series([train.mean()] * test_size, index=test.index)
            baseline_mae = mean_absolute_error(test, baseline_pred)
            baseline_mape = calculate_safe_mape(test, baseline_pred)

            results_summary.append({
                "District": district, "Frequency": freq_name, "Model_Type": "Baseline",
                "Params": "Mean",
                "MAE": baseline_mae,
                "MAPE": baseline_mape,
                "Improvement_Pct": 0.0
            })

            for m_name, m_func, m_args in models_to_run:
                try:
                    model_obj, params_str = m_func(*m_args)
                    if model_obj:
                        pred = model_obj.forecast(test_size)
                        if hasattr(pred, "predicted_mean"): pred = pred.predicted_mean
                        pred = pd.Series(pred.values, index=test.index).clip(lower=0)

                        mae = mean_absolute_error(test, pred)
                        mape = calculate_safe_mape(test, pred)

                        if mae < best_mae:
                            best_mae = mae
                            winner_name = f"{m_name} ({params_str})"
                            winner_preds = pred

                        imp = ((baseline_mae - mae) / baseline_mae) * 100

                        results_summary.append({
                            "District": district, "Frequency": freq_name, "Model_Type": m_name,
                            "Params": params_str,
                            "MAE": mae,
                            "MAPE": mape,
                            "Improvement_Pct": imp
                        })
                except:
                    continue

            # --- PLOT WINNER ---
            if winner_preds is not None:
                plt.figure(figsize=(12, 6))

                lookback = max(test_size * 4, 20)
                if len(train) > lookback:
                    plot_start = train.index[-lookback]
                else:
                    plot_start = train.index[0]

                # Calculate winner mape for title
                win_mape_title = calculate_safe_mape(test, winner_preds)

                plt.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='History', color='gray', alpha=0.5)
                plt.plot(test.index, test, label='Actual', marker='o', color='black')
                plt.plot(winner_preds.index, winner_preds,
                         label=f'WINNER: {winner_name}\nMAE={best_mae:.2f} | MAPE={win_mape_title:.1f}%',
                         color='green', linestyle='--', marker='x')
                plt.plot(baseline_pred.index, baseline_pred, label=f'Baseline (MAE={baseline_mae:.2f})',
                         color='blue', linestyle=':')

                plt.title(f"{district} ({freq_name}): Best Forecast Model")
                plt.legend()
                plt.grid(True, alpha=0.3)

                safe_name = dist_norm.replace(" ", "_").replace("İ", "I").replace("Ş", "S").replace("Ç", "C").replace(
                    "Ğ", "G").replace("Ö", "O").replace("Ü", "U")
                plt.savefig(f"district_analysis_plots/{safe_name}_{freq_name}_Best.png")
                plt.close()

    # --- SAVE RESULTS ---
    res_df = pd.DataFrame(results_summary)

    # Reorder columns to put MAPE near MAE
    cols = ['District', 'Frequency', 'Model_Type', 'Params', 'MAE', 'MAPE', 'Improvement_Pct']
    res_df = res_df[cols]

    res_df.to_csv("district_analysis_detailed.csv", index=False)

    winners = res_df.loc[res_df.groupby(['District', 'Frequency'])['MAE'].idxmin()]
    winners.to_csv("district_analysis_winners.csv", index=False)

    print("\n" + "=" * 50)
    print("DISTRICT ANALYSIS COMPLETE")
    print("Top Models per District/Frequency:")
    print(winners[['District', 'Frequency', 'Model_Type', 'MAE', 'MAPE', 'Improvement_Pct']].head(10))


if __name__ == "__main__":
    run_analysis()