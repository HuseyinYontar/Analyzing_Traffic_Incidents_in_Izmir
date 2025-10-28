#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# Dosya Yolları
# =========================
ACC_PATH   = Path("izbb-kaza-ariza-verileri_with_ilce_updated.xlsx")
WX_GAZ_CSV = Path("../weatherData/izmir_weather_2021-12-01_to_2025-09-28_Gaziemir.csv")
WX_CIG_CSV = Path("../weatherData/izmir_weather_2021-12-01_to_2025-09-28_Cigli.csv")

# =========================
# Parametreler
# =========================
GAZ_BEFORE_MIN = 30
GAZ_AFTER_MIN  = 30
CIG_TOL_MIN    = 10  # 0 yaparsan sadece tam :50

OUT_XLSX = ACC_PATH.with_name(ACC_PATH.stem + "_with_weather.xlsx")
OUT_CSV  = ACC_PATH.with_name(ACC_PATH.stem + "_with_weather.csv")

# =========================
# Yardımcılar
# =========================
def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _parse_with_known_formats(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """
    WU CSV'lerde genelde:
      - Date: YYYY-MM-DD
      - Time: HH:MM  veya  HH:MM:SS
    Bu koşulları kontrol edip to_datetime'a net format veriyoruz.
    Uymayan satırlar için fallback var (dayfirst=True).
    """
    date_s = date_s.astype(str).str.strip()
    time_s = time_s.astype(str).str.strip()
    combo = date_s + " " + time_s

    date_ok = date_s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    time_hm = time_s.str.match(r"^\d{2}:\d{2}$", na=False)
    time_hms= time_s.str.match(r"^\d{2}:\d{2}:\d{2}$", na=False)

    # Tümü aynı desendeyse tek shot parse
    if date_ok.all() and time_hms.all():
        return pd.to_datetime(combo, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if date_ok.all() and time_hm.all():
        return pd.to_datetime(combo, format="%Y-%m-%d %H:%M", errors="coerce")

    # Karışık/bozuk kayıtlar varsa vektörel fallback (coerce + dayfirst)
    return pd.to_datetime(combo, errors="coerce", dayfirst=True)

def _standardize_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_cols(df)

    date_col = next((c for c in ["Date","DATE","Tarih","TARIH"] if c in df.columns), None)
    time_col = next((c for c in ["Time","TIME","Saat","SAAT"] if c in df.columns), None)
    if date_col is None or time_col is None:
        raise ValueError("Weather CSV'de 'Date' ve 'Time' benzeri kolonlar bulunamadı.")

    # >>> UYARIYI SUSTURAN KISIM: format belirli
    df["wx_ts"] = _parse_with_known_formats(df[date_col], df[time_col])

    # Kolon isimlerini normalize et
    rename_map = {}
    for col in df.columns:
        lc = str(col).lower().replace(" ", "").replace("_", "")
        if lc in ("temperaturec","temp_c","temperature"):
            rename_map[col] = "Temp_C"
        elif lc in ("humidity","relativehumidity","humiditypercent"):
            rename_map[col] = "Humidity"
        elif lc in ("windspeedkmh","windspeed(km/h)","windspeedkm/h","windspeedkph","windkmh"):
            rename_map[col] = "Wind_kmh"
        elif lc in ("precipitationmm","precipmm","precipitation","rainmm"):
            rename_map[col] = "Precip_mm"
        elif lc in ("conditions","condition","weather"):
            rename_map[col] = "Conditions"

    df = df.rename(columns=rename_map)

    keep = ["wx_ts","Temp_C","Humidity","Wind_kmh","Precip_mm","Conditions"]
    keep = [k for k in keep if k in df.columns]

    df = (
        df[keep]
        .dropna(subset=["wx_ts"])
        .drop_duplicates(subset=["wx_ts"])
        .sort_values("wx_ts")
        .reset_index(drop=True)
    )
    # Tipleri düzelt (numarik olanları float'a çevir)
    for num in ["Temp_C","Humidity","Wind_kmh","Precip_mm"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")
    return df

def _build_kaza_ts(row, date_cols=("TARIH","Tarih","DATE","Date"),
                   time_cols=("KAZA_ZAMANI","Kaza_Zamani","SAAT","Saat","TIME","Time")):
    date_val = None
    time_val = None
    for c in date_cols:
        if c in row and pd.notna(row[c]):
            date_val = str(row[c]).strip()
            break
    for c in time_cols:
        if c in row and pd.notna(row[c]):
            time_val = str(row[c]).strip()
            break
    if date_val is None and time_val is None:
        for c in row.index:
            lc = str(c).lower()
            if ("tarih" in lc or "date" in lc) and pd.notna(row[c]):
                return pd.to_datetime(row[c], errors="coerce", dayfirst=True)
        return pd.NaT
    if time_val is None:
        return pd.to_datetime(date_val, errors="coerce", dayfirst=True)
    # Burada farklı biçimler olabilir; basit birleşim coerce ile güvenli
    return pd.to_datetime(f"{date_val} {time_val}", errors="coerce", dayfirst=True)

def _norm_ilce(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = (s.replace("ı","i").replace("ğ","g").replace("ç","c")
           .replace("ö","o").replace("ş","s").replace("ü","u"))
    return s

def _lookup_gaziemir(wx_df: pd.DataFrame, ts: pd.Timestamp, before_min=GAZ_BEFORE_MIN, after_min=GAZ_AFTER_MIN):
    if pd.isna(ts) or wx_df.empty:
        return None
    times = wx_df["wx_ts"].values
    idx = np.searchsorted(times, np.datetime64(ts))

    best = None
    best_delta = None

    if idx - 1 >= 0:
        t = pd.Timestamp(times[idx-1])
        d = (ts - t).total_seconds() / 60.0
        if d >= 0 and d <= before_min:
            best = idx - 1
            best_delta = d

    if idx < len(times):
        t = pd.Timestamp(times[idx])
        d = (t - ts).total_seconds() / 60.0
        if d >= 0 and d <= after_min:
            if best_delta is None or d < best_delta:
                best = idx
                best_delta = d

    if best is None:
        return None

    row = wx_df.iloc[best]
    return {
        "wx_ts": row.get("wx_ts", pd.NaT),
        "Temp_C": row.get("Temp_C", np.nan),
        "Humidity": row.get("Humidity", np.nan),
        "Wind_kmh": row.get("Wind_kmh", np.nan),
        "Precip_mm": row.get("Precip_mm", np.nan),
        "Conditions": row.get("Conditions", np.nan),
        "_delta_min": best_delta,
    }

def _snap_cigli_target(ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.minute > 25:
        return ts.replace(minute=50, second=0, microsecond=0)
    else:
        one_hour_before = ts - pd.Timedelta(hours=1)
        return one_hour_before.replace(minute=50, second=0, microsecond=0)

def _lookup_exact_or_tol(wx_df: pd.DataFrame, target_ts: pd.Timestamp, tol_min=CIG_TOL_MIN):
    if pd.isna(target_ts) or wx_df.empty:
        return None
    times = wx_df["wx_ts"].values
    idx = np.searchsorted(times, np.datetime64(target_ts))

    candidates = []
    for j in (idx-1, idx, idx+1):
        if 0 <= j < len(times):
            candidates.append(j)

    best = None
    best_delta = None
    for j in candidates:
        t = pd.Timestamp(times[j])
        delta = abs((t - target_ts).total_seconds()) / 60.0
        ok = (tol_min is None) or (tol_min >= 0 and delta <= tol_min) or (tol_min < 0 and delta == 0)
        if ok and (best_delta is None or delta < best_delta):
            best = j
            best_delta = delta

    if best is None:
        return None

    row = wx_df.iloc[best]
    return {
        "wx_ts": row.get("wx_ts", pd.NaT),
        "Temp_C": row.get("Temp_C", np.nan),
        "Humidity": row.get("Humidity", np.nan),
        "Wind_kmh": row.get("Wind_kmh", np.nan),
        "Precip_mm": row.get("Precip_mm", np.nan),
        "Conditions": row.get("Conditions", np.nan),
        "_delta_min": best_delta,
    }

# =========================
# Yükle
# =========================
acc = pd.read_excel(ACC_PATH)
acc = _strip_cols(acc)

if   "ILCE" in acc.columns:  ilce_col = "ILCE"
elif "İLÇE" in acc.columns:  ilce_col = "İLÇE"
elif "ilce" in acc.columns:  ilce_col = "ilce"
else:
    raise ValueError("Kaza dosyasında 'ILCE/İLÇE/ilce' kolonu bulunamadı.")

wx_gaz = _standardize_weather(pd.read_csv(WX_GAZ_CSV))
wx_cig = _standardize_weather(pd.read_csv(WX_CIG_CSV))

# =========================
# Çıktı DF'yi DOĞRU DTYPE ile hazırla (FutureWarning fix)
# =========================
out = acc.copy()

# datetime kolonları
for dcol in ["kaza_ts", "wx_ts"]:
    if dcol not in out.columns:
        out[dcol] = pd.NaT
    out[dcol] = pd.to_datetime(out[dcol])  # dtype: datetime64[ns]

# numerik kolonlar
for ncol in ["Temp_C","Humidity","Wind_kmh","Precip_mm","WEATHER_DELTA_MIN"]:
    if ncol not in out.columns:
        out[ncol] = np.nan
    out[ncol] = pd.to_numeric(out[ncol], errors="coerce")  # dtype: float64

# metin kolonları
for scol in ["Conditions","WEATHER_MATCH","WEATHER_SOURCE"]:
    if scol not in out.columns:
        out[scol] = pd.Series([None]*len(out), dtype="object")
    else:
        out[scol] = out[scol].astype("object")

# =========================
# Satır Satır İşle
# =========================
for i, r in out.iterrows():
    # kaza zamanı (datetime64[ns] sütuna yaz)
    kts = _build_kaza_ts(r)
    out.at[i, "kaza_ts"] = pd.to_datetime(kts) if pd.notna(kts) else pd.NaT

    ilce_norm = _norm_ilce(str(r.get(ilce_col, "")))

    if ilce_norm == "gaziemir":
        out.at[i, "WEATHER_SOURCE"] = "gaziemir"
        match = _lookup_gaziemir(wx_gaz, kts, GAZ_BEFORE_MIN, GAZ_AFTER_MIN)

    elif ilce_norm == "cigli":  # çiğli
        out.at[i, "WEATHER_SOURCE"] = "cigli"
        target = _snap_cigli_target(kts)
        match  = _lookup_exact_or_tol(wx_cig, target, CIG_TOL_MIN)

    else:
        continue  # diğer ilçeler

    if match is None:
        continue

    # datetime hedef kolona datetime, string hedef kolona object, sayısal hedefe float yazıyoruz
    out.at[i, "wx_ts"]      = pd.to_datetime(match["wx_ts"]) if pd.notna(match["wx_ts"]) else pd.NaT
    out.at[i, "Temp_C"]     = float(match["Temp_C"])   if pd.notna(match["Temp_C"])   else np.nan
    out.at[i, "Humidity"]   = float(match["Humidity"]) if pd.notna(match["Humidity"]) else np.nan
    out.at[i, "Wind_kmh"]   = float(match["Wind_kmh"]) if pd.notna(match["Wind_kmh"]) else np.nan
    out.at[i, "Precip_mm"]  = float(match["Precip_mm"])if pd.notna(match["Precip_mm"])else np.nan
    out.at[i, "Conditions"] = None if pd.isna(match["Conditions"]) else str(match["Conditions"])
    out.at[i, "WEATHER_DELTA_MIN"] = float(match["_delta_min"]) if match["_delta_min"] is not None else np.nan

    parts = []
    for c in ["Temp_C","Humidity","Wind_kmh","Precip_mm","Conditions"]:
        v = match[c]
        if pd.notna(v):
            parts.append(f"{c}={v}")
    out.at[i, "WEATHER_MATCH"] = (f"[t={match['wx_ts']}] " + ", ".join(parts)) if parts else (f"[t={match['wx_ts']}]")

# =========================
# Kaydet & Kontrol
# =========================
out.to_excel(OUT_XLSX, index=False)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
assert len(out) == len(acc), f"Row count changed! expected={len(acc)}, got={len(out)}"

print("Tamamlandı.")
print(f"Excel çıktı: {OUT_XLSX}")
print(f"CSV çıktı  : {OUT_CSV}")
