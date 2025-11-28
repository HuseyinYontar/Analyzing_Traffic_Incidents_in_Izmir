import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

# Scikit-Learn Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
    'Daily': {'freq': 'D', 'test_size': 30},
    'Weekly': {'freq': 'W-MON', 'test_size': 12},
    'Monthly': {'freq': 'M', 'test_size': 6}
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


# --- 3. FEATURE ENGINEERING ---
def create_features(ts_data, freq_name):
    """
    Creates deterministic features (Calendar + Trend) from the index.
    Allows for forecasting without recursive lag updates.
    """
    df_feat = pd.DataFrame(index=ts_data.index)
    df_feat['y'] = ts_data.values

    # 1. Trend Feature (Time Index)
    df_feat['TimeIndex'] = np.arange(len(ts_data))

    # 2. Calendar Features
    df_feat['Month'] = df_feat.index.month
    df_feat['Year'] = df_feat.index.year
    df_feat['Quarter'] = df_feat.index.quarter

    if freq_name == 'Daily':
        df_feat['DayOfWeek'] = df_feat.index.dayofweek
        df_feat['DayOfMonth'] = df_feat.index.day
        df_feat['WeekOfYear'] = df_feat.index.isocalendar().week.astype(int)
        df_feat['IsWeekend'] = (df_feat['DayOfWeek'] >= 5).astype(int)
    elif freq_name == 'Weekly':
        df_feat['WeekOfYear'] = df_feat.index.isocalendar().week.astype(int)
        # Week of year captures seasonality well for weekly data
    elif freq_name == 'Monthly':
        # For monthly, Month is the primary seasonal feature
        pass

    return df_feat


# --- 4. MODEL SUITE ---
def get_regressors():
    """Returns a dictionary of models to test."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                                      random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))  # SVR requires scaling
    }


# --- 5. MAIN ANALYSIS ROUTINE ---
def run_regression_analysis():
    df = load_data()
    if df is None: return

    results_summary = []

    if not os.path.exists("regression_plots"):
        os.makedirs("regression_plots")

    print(f"\nStarting Regression Analysis for {len(TARGET_STREETS)} streets...")

    for freq_name, freq_config in FREQUENCIES.items():
        print(f"\n=== Processing {freq_name} Data ===")

        for street in TARGET_STREETS:
            print(f"  Analyzing {street}...")

            # --- DATA PREP (Same logic as before) ---
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

            # --- CREATE FEATURES ---
            df_features = create_features(ts, freq_name)

            # Split X (Features) and y (Target)
            X = df_features.drop(columns=['y'])
            y = df_features['y']

            # Train/Test Split
            X_train = X.iloc[:-test_size]
            X_test = X.iloc[-test_size:]
            y_train = y.iloc[:-test_size]
            y_test = y.iloc[-test_size:]

            # --- RUN MODELS ---
            models = get_regressors()

            best_mae = float("inf")
            winner_name = "None"
            winner_preds = None

            # 1. Baseline (Dummy Regressor)
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(X_train, y_train)
            dummy_pred = dummy.predict(X_test)
            baseline_mae = mean_absolute_error(y_test, dummy_pred)

            # Store Baseline
            results_summary.append({
                "Street": street, "Frequency": freq_name, "Model": "DummyRegressor",
                "MAE": baseline_mae, "Improvement_Pct": 0.0
            })

            # 2. Iterate Regressors
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    # Clip predictions to 0 (cannot have negative incidents)
                    pred = pd.Series(pred, index=y_test.index).clip(lower=0)

                    mae = mean_absolute_error(y_test, pred)

                    # Check Winner
                    if mae < best_mae:
                        best_mae = mae
                        winner_name = name
                        winner_preds = pred

                    imp = ((baseline_mae - mae) / baseline_mae) * 100
                    results_summary.append({
                        "Street": street, "Frequency": freq_name, "Model": name,
                        "MAE": mae, "Improvement_Pct": imp
                    })
                except Exception as e:
                    continue

            # --- PLOTTING ---
            if winner_preds is not None:
                plt.figure(figsize=(12, 6))
                plot_start = y_train.index[-max(test_size * 4, 20)]

                plt.plot(y_train.loc[plot_start:].index, y_train.loc[plot_start:], label='Historical', color='gray',
                         alpha=0.5)
                plt.plot(y_test.index, y_test, label='Actual', marker='o', color='black', linewidth=1.5)

                # Plot Winner
                plt.plot(winner_preds.index, winner_preds, label=f'WINNER: {winner_name}\nMAE={best_mae:.2f}',
                         color='purple', linestyle='--', marker='s', linewidth=2)

                # Plot Baseline
                dummy_series = pd.Series(dummy_pred, index=y_test.index)
                plt.plot(dummy_series.index, dummy_series, label=f'Baseline (Mean)\nMAE={baseline_mae:.2f}',
                         color='blue', linestyle=':', alpha=0.7)

                plt.title(f"{street} ({freq_name}): Regression Forecast (Feature-Based)")
                plt.ylabel("Incidents")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)

                safe_name = street.replace(" ", "_").replace("İ", "I").replace("Ş", "S").replace("Ç", "C")
                plt.savefig(f"regression_plots/{safe_name}_{freq_name}_Reg.png")
                plt.close()

    # --- REPORTING ---
    res_df = pd.DataFrame(results_summary)
    res_df.to_csv("street_regression_detailed.csv", index=False)

    # Winners Summary
    winners = res_df.loc[res_df.groupby(['Street', 'Frequency'])['MAE'].idxmin()]
    winners.to_csv("street_regression_winners.csv", index=False)

    print("\n" + "=" * 50)
    print("REGRESSION ANALYSIS COMPLETE")
    print("=" * 50)
    print("Top Performing Regressors per Street/Frequency:")
    print(winners[['Street', 'Frequency', 'Model', 'MAE', 'Improvement_Pct']].head(10))
    print("\nDetailed results saved to 'street_regression_detailed.csv'")
    print("Winner summary saved to 'street_regression_winners.csv'")


if __name__ == "__main__":
    run_regression_analysis()