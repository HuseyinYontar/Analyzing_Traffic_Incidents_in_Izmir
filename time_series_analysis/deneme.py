import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import warnings
import itertools
import os

# --- 1. CONFIGURATION & IMPORTS ---
warnings.filterwarnings("ignore")

TARGET_STREETS = [
    "YEŞILDERE CADDESI",
    "ANADOLU CADDESI",
    "MÜRSELPAŞA BULVARI",
    "AKÇAY CADDESI",
    "MUSTAFA KEMAL SAHIL BULVARI",
    "ANKARA CADDESI",
    "ALTINYOL CADDESI",
    "YEŞILLIK CADDESI"
]

# We will filter for these, but count them TOGETHER
TARGET_ACCIDENT_TYPES = [
    "MADDI HASARLI",
    "YARALANMALI/ÖLÜMLÜ"
]

FREQUENCIES = {
    'Daily': {'freq': 'D', 'period': 7, 'test_size': 30},
    'Weekly': {'freq': 'W-MON', 'period': 52, 'test_size': 12},
    'Monthly': {'freq': 'M', 'period': 12, 'test_size': 6}
}


# --- 2. DATA LOADING ---
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
        except Exception as e:
            print(f"Error loading local file: {e}")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Clean Columns
    df.columns = df.columns.str.strip().str.upper()

    # Process Dates
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

    # Clean String Columns
    if "CADDE" in df.columns:
        df["CADDE"] = df["CADDE"].str.strip().str.upper()

    if "KAZA_TIPI" in df.columns:
        df["KAZA_TIPI"] = df["KAZA_TIPI"].str.strip().str.upper()

    # 1. Filter Streets
    df = df[df["CADDE"].isin(TARGET_STREETS)]

    # 2. Filter Accident Types (Keep only the 2 targets)
    # Note: We filter rows here, but we won't group by this column later.
    # This effectively removes any "Unknown" or other types from the count.
    df = df[df["KAZA_TIPI"].isin(TARGET_ACCIDENT_TYPES)]

    return df


# --- 3. MODEL FITTING FUNCTIONS ---

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
        {'trend': 'add', 'seasonal': None},  # Double ES
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': period},  # Triple ES
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': period},
        {'trend': None, 'seasonal': None}  # Simple ES
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


def calculate_metrics(y_true, y_pred):
    """Calculates MAE and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE handling for zeros
    try:
        if (y_true == 0).any():
            # Add tiny epsilon to avoid division by zero errors
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
        else:
            mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = float('nan')

    return mae, mape


# --- 4. MAIN ANALYSIS ROUTINE ---
def run_analysis():
    df = load_data()
    if df is None or df.empty:
        print("No data found or data is empty after filtering.")
        return

    results_summary = []

    if not os.path.exists("analysis_plots"):
        os.makedirs("analysis_plots")

    print(f"\nStarting Combined Analysis for {len(TARGET_STREETS)} streets...")
    print(f"Including only: {TARGET_ACCIDENT_TYPES}")

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n=== Processing {freq_name} Data ===")

        for street in TARGET_STREETS:
            print(f"  Analyzing {street} (Combined Types)...")

            # Filter by Street
            street_data = df[df["CADDE"] == street]

            if street_data.empty:
                print(f"    Skipping: No data for {street}")
                continue

            # --- PREPARE TIME SERIES ---
            # KEY CHANGE: We do NOT loop through types here.
            # We simply resample the dataframe. Since we already filtered the DF
            # to only include the 2 target types in load_data(),
            # .size() counts the TOTAL combined incidents.
            ts_resampled = street_data.set_index("TARIH").resample(freq_config['freq']).size()

            # Handle Missing Data / Zeros
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

            # --- DEFINE MODEL SUITE ---
            models_to_run = []
            models_to_run.append(("AR", fit_sarimax_grid, (train, range(1, 4), [0], [0], [(0, 0, 0, 0)])))
            models_to_run.append(("MA", fit_sarimax_grid, (train, [0], [0], range(1, 4), [(0, 0, 0, 0)])))
            models_to_run.append(("ARMA", fit_sarimax_grid, (train, range(1, 3), [0], range(1, 3), [(0, 0, 0, 0)])))
            models_to_run.append(
                ("ARIMA", fit_sarimax_grid, (train, range(0, 3), range(0, 2), range(0, 3), [(0, 0, 0, 0)])))

            use_seasonal = len(train) >= (2 * period)
            s_grid = [(0, 0, 0, 0)]
            if use_seasonal and period > 1:
                s_grid = [(1, 0, 0, period), (0, 1, 0, period), (1, 1, 1, period), (0, 0, 0, 0)]
            models_to_run.append(("SARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), s_grid)))
            models_to_run.append(("ExpSmoothing", get_exponential_smoothing, (train, period)))

            # --- RUN MODELS ---
            best_mae_for_street = float("inf")
            winner_model_name = "None"
            winner_preds = None
            winner_mape = None

            # Baseline
            baseline_pred = pd.Series([train.mean()] * test_size, index=test.index)
            baseline_mae, baseline_mape = calculate_metrics(test, baseline_pred)

            results_summary.append({
                "Street": street, "Type_Group": "Combined", "Frequency": freq_name,
                "Model_Type": "Baseline", "Params": "Mean",
                "MAE": baseline_mae, "MAPE": baseline_mape, "Improvement_Pct": 0.0
            })

            for m_name, m_func, m_args in models_to_run:
                try:
                    model_obj, params_str = m_func(*m_args)

                    if model_obj:
                        pred = model_obj.forecast(test_size)
                        if hasattr(pred, "predicted_mean"):
                            pred = pred.predicted_mean

                        pred = pd.Series(pred.values, index=test.index).clip(lower=0)

                        mae, mape = calculate_metrics(test, pred)

                        if mae < best_mae_for_street:
                            best_mae_for_street = mae
                            winner_model_name = f"{m_name} ({params_str})"
                            winner_preds = pred
                            winner_mape = mape

                        imp = 0
                        if baseline_mae > 0:
                            imp = ((baseline_mae - mae) / baseline_mae) * 100

                        results_summary.append({
                            "Street": street, "Type_Group": "Combined", "Frequency": freq_name,
                            "Model_Type": m_name, "Params": params_str,
                            "MAE": mae, "MAPE": mape, "Improvement_Pct": imp
                        })
                except Exception as e:
                    continue

            # --- PLOTTING ---
            if winner_preds is not None:
                plt.figure(figsize=(12, 6))
                plot_start = train.index[-max(test_size * 4, 20)]

                plt.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='Historical', color='gray',
                         alpha=0.5)
                plt.plot(test.index, test, label='Actual', marker='o', color='black', linewidth=1.5)

                plt.plot(winner_preds.index, winner_preds,
                         label=f'WINNER: {winner_model_name}\nMAE={best_mae_for_street:.2f} | MAPE={winner_mape:.2f}',
                         color='green', linestyle='--', marker='x', linewidth=2)

                plt.plot(baseline_pred.index, baseline_pred,
                         label=f'Baseline (Mean)\nMAE={baseline_mae:.2f}',
                         color='blue', linestyle=':', alpha=0.7)

                plt.title(f"{street} ({freq_name}): Combined Accident Forecast")
                plt.ylabel("Total Incidents (Damage + Injury/Fatal)")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)

                safe_street = street.replace(" ", "_").replace("İ", "I").replace("Ş", "S").replace("Ç", "C")
                plt.savefig(f"analysis_plots/{safe_street}_COMBINED_{freq_name}_BestFit.png")
                plt.close()

    # --- REPORTING ---
    if not results_summary:
        print("No results generated.")
        return

    res_df = pd.DataFrame(results_summary)
    res_df.to_csv("street_combined_analysis_detailed.csv", index=False)

    winners = res_df.loc[res_df.groupby(['Street', 'Frequency'])['MAE'].idxmin()]
    winners.to_csv("street_combined_analysis_winners.csv", index=False)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Top Performing Models (Combined Data):")
    print(winners[['Street', 'Frequency', 'Model_Type', 'MAE', 'MAPE', 'Improvement_Pct']].head(10))


if __name__ == "__main__":
    run_analysis()