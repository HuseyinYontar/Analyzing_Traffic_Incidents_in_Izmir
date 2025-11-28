import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    df.columns = df.columns.str.strip().str.upper()
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df = df[df["CADDE"].str.strip().str.upper().isin(TARGET_STREETS)]
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
    # Try different configurations
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
        # Skip seasonal configs if period is invalid or data too short
        if cfg['seasonal'] is not None and (period < 2 or len(train) < 2 * period):
            continue

        try:
            model = ExponentialSmoothing(train,
                                         trend=cfg['trend'],
                                         seasonal=cfg['seasonal'],
                                         seasonal_periods=cfg.get('seasonal_periods')).fit()
            # ES doesn't strictly have AIC in the same way statsmodels wrapper returns it compared to ARIMA
            # consistently across versions, but .aic attribute usually exists.
            # If not, we use SSE as proxy for "goodness" on training data.
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

    if not os.path.exists("analysis_plots"):
        os.makedirs("analysis_plots")

    print(f"\nStarting Analysis for {len(TARGET_STREETS)} streets...")

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n=== Processing {freq_name} Data ===")

        for street in TARGET_STREETS:
            print(f"  Analyzing {street}...")

            # --- PREPARE DATA ---
            street_data = df[df["CADDE"].str.strip().str.upper() == street]
            ts_resampled = street_data.set_index("TARIH").resample(freq_config['freq']).size()

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
            # Format: (Type_Name, Function, Args)
            models_to_run = []

            # 1. AR (p=1..3, d=0, q=0)
            models_to_run.append(("AR", fit_sarimax_grid, (train, range(1, 4), [0], [0], [(0, 0, 0, 0)])))

            # 2. MA (p=0, d=0, q=1..3)
            models_to_run.append(("MA", fit_sarimax_grid, (train, [0], [0], range(1, 4), [(0, 0, 0, 0)])))

            # 3. ARMA (p=1..2, d=0, q=1..2)
            models_to_run.append(("ARMA", fit_sarimax_grid, (train, range(1, 3), [0], range(1, 3), [(0, 0, 0, 0)])))

            # 4. ARIMA (p=0..2, d=0..1, q=0..2)
            models_to_run.append(
                ("ARIMA", fit_sarimax_grid, (train, range(0, 3), range(0, 2), range(0, 3), [(0, 0, 0, 0)])))

            # 5. SARIMA (Standard grid, with seasonality)
            use_seasonal = len(train) >= (2 * period)
            s_grid = [(0, 0, 0, 0)]
            if use_seasonal and period > 1:
                # Simplified seasonal grid to save time
                s_grid = [(1, 0, 0, period), (0, 1, 0, period), (1, 1, 1, period), (0, 0, 0, 0)]
            models_to_run.append(("SARIMA", fit_sarimax_grid, (train, range(0, 2), range(0, 2), range(0, 2), s_grid)))

            # 6. Exponential Smoothing
            models_to_run.append(("ExpSmoothing", get_exponential_smoothing, (train, period)))

            # --- RUN MODELS ---
            best_mae_for_street = float("inf")
            winner_model_name = "None"
            winner_preds = None

            baseline_pred = pd.Series([train.mean()] * test_size, index=test.index)
            baseline_mae = mean_absolute_error(test, baseline_pred)

            # Store baseline first
            results_summary.append({
                "Street": street, "Frequency": freq_name, "Model_Type": "Baseline",
                "Params": "Mean", "MAE": baseline_mae, "Improvement_Pct": 0.0
            })

            for m_name, m_func, m_args in models_to_run:
                try:
                    model_obj, params_str = m_func(*m_args)

                    if model_obj:
                        pred = model_obj.forecast(test_size)
                        # Handle different return types (ES returns Series, SARIMAX wrapper needs .predicted_mean sometimes or direct)
                        if hasattr(pred, "predicted_mean"):
                            pred = pred.predicted_mean

                        # Align index explicitly
                        pred = pd.Series(pred.values, index=test.index).clip(lower=0)

                        mae = mean_absolute_error(test, pred)

                        # Check if winner
                        if mae < best_mae_for_street:
                            best_mae_for_street = mae
                            winner_model_name = f"{m_name} ({params_str})"
                            winner_preds = pred

                        # Log Result
                        imp = ((baseline_mae - mae) / baseline_mae) * 100
                        results_summary.append({
                            "Street": street, "Frequency": freq_name, "Model_Type": m_name,
                            "Params": params_str, "MAE": mae, "Improvement_Pct": imp
                        })
                except Exception as e:
                    # print(f"  Error fitting {m_name}: {e}") # specific debug
                    continue

            # --- PLOTTING (Winner vs Baseline) ---
            if winner_preds is not None:
                plt.figure(figsize=(12, 6))
                plot_start = train.index[-max(test_size * 4, 20)]

                plt.plot(train.loc[plot_start:].index, train.loc[plot_start:], label='Historical', color='gray',
                         alpha=0.5)
                plt.plot(test.index, test, label='Actual', marker='o', color='black', linewidth=1.5)

                # Plot Winner
                plt.plot(winner_preds.index, winner_preds,
                         label=f'WINNER: {winner_model_name}\nMAE={best_mae_for_street:.2f}',
                         color='green', linestyle='--', marker='x', linewidth=2)

                # Plot Baseline
                plt.plot(baseline_pred.index, baseline_pred, label=f'Baseline (Mean)\nMAE={baseline_mae:.2f}',
                         color='blue', linestyle=':', alpha=0.7)

                plt.title(f"{street} ({freq_name}): Best Model Forecast")
                plt.ylabel("Incidents")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)

                safe_name = street.replace(" ", "_").replace("İ", "I").replace("Ş", "S").replace("Ç", "C")
                plt.savefig(f"analysis_plots/{safe_name}_{freq_name}_BestFit.png")
                plt.close()

    # --- REPORTING ---
    res_df = pd.DataFrame(results_summary)
    res_df.to_csv("street_analysis_detailed.csv", index=False)

    # Create a concise summary of just the winners
    winners = res_df.loc[res_df.groupby(['Street', 'Frequency'])['MAE'].idxmin()]
    winners.to_csv("street_analysis_winners.csv", index=False)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print("Top Performing Models per Street/Frequency:")
    print(winners[['Street', 'Frequency', 'Model_Type', 'MAE', 'Improvement_Pct']].head(10))
    print("\nDetailed results saved to 'street_analysis_detailed.csv'")
    print("Winner summary saved to 'street_analysis_winners.csv'")


if __name__ == "__main__":
    run_analysis()