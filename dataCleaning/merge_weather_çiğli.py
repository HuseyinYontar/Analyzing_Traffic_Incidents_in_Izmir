import pandas as pd

# File paths
acc_path = "izbb-kaza-ariza-verileri_with_ilce_updated_with_weather.xlsx"
wx_path  = "merged_accidents_weather.xlsx"
out_path = "accidents_merged_with_temperature_condition.xlsx"

# Load
acc = pd.read_excel(acc_path)
wx  = pd.read_excel(wx_path)

# Normalize join keys: TARIH (date), ILCE (trim), KAZA_ZAMANI (24h time with seconds)
def _norm_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _norm_time(s):
    # Handles '14:48:00', '2:48 PM', etc., and returns 'HH:MM:SS'
    t = pd.to_datetime(s, errors="coerce")
    return t.dt.strftime("%H:%M:%S")

def _norm_text(s):
    return s.astype(str).str.strip()

# Prepare keys on accidents
acc["_TARIH_KEY"] = _norm_date(acc["TARIH"])
acc["_TIME_KEY"]  = _norm_time(acc["KAZA_ZAMANI"])
acc["_ILCE_KEY"]  = _norm_text(acc["ILCE"])

# Prepare keys on weather file
wx["_TARIH_KEY"] = _norm_date(wx["TARIH"])
wx["_TIME_KEY"]  = _norm_time(wx["KAZA_ZAMANI"])
wx["_ILCE_KEY"]  = _norm_text(wx["ILCE"])

# Columns to bring from weather file (only if present)
keep_cols = ["Temperature", "Condition"]
present_keep = [c for c in keep_cols if c in wx.columns]

# Merge (left: accidents)
merged = acc.merge(
    wx[["_TARIH_KEY", "_TIME_KEY", "_ILCE_KEY"] + present_keep],
    on=["_TARIH_KEY", "_TIME_KEY", "_ILCE_KEY"],
    how="left",
    suffixes=("", "_wx")
)

# Drop helper keys
merged = merged.drop(columns=["_TARIH_KEY", "_TIME_KEY", "_ILCE_KEY"])

# Save
merged.to_excel(out_path, index=False)

print(f"Saved -> {out_path}")
