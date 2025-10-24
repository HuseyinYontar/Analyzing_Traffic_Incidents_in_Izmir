import pandas as pd
import numpy as np

in_path = "izbb-kaza-ariza-verileri-cleaned-direction-type-street.xlsx"
df = pd.read_excel(in_path)

base_date = pd.to_datetime(df["TARIH"], errors="coerce").dt.normalize()


kaza_td = pd.to_timedelta(df["KAZA_ZAMANI"].astype(str), errors="coerce")
mud_td  = pd.to_timedelta(df["MUDAHALE_ZAMANI"].astype(str), errors="coerce")

kaza_dt = base_date + kaza_td
mud_dt  = base_date + mud_td

delta = mud_dt - kaza_dt
neg = delta.dt.total_seconds() < 0

days_to_add = np.zeros(len(df), dtype=int)
days_to_add[neg] = np.ceil(
    (-delta.dt.total_seconds()[neg]) / (24 * 3600)
).astype(int)

mud_dt = mud_dt + pd.to_timedelta(days_to_add, unit="D")

delta = mud_dt - kaza_dt

elapsed_minutes = delta.dt.total_seconds() / 60.0

missing_mask = kaza_dt.isna() | mud_dt.isna()
elapsed_minutes[missing_mask] = np.nan

df["MUDAHALE_SURESI_DK"] = elapsed_minutes


out_path = "izbb-kaza-ariza-verileri_cleaned.xlsx"
df.to_excel(out_path, index=False)
print(f"✅ Saved: {out_path}")
