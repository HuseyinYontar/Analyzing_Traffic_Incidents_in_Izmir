import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo  # Python 3.9+

import pandas as pd
import requests

API_KEY   = "e1f10a1e78da46f5b10a1e78da96f525"  # <-- your key
LOCATION  = "LTBJ:9:TR"# İzmir Adnan Menderes (use LTBL:9:TR if you prefer that station)
#LOCATION = "LTBL:9:TR" #Çiğli
UNITS     = "m"          # m = metric (°C, km/h, mb); use 'e' for imperial
TZ_LOCAL  = ZoneInfo("Europe/Istanbul")
BASE_URL  = "https://api.weather.com/v1/location/{loc}/observations/historical.json"

START = date(2021, 12, 1)
END   = date(2025, 9, 28)  # inclusive



def fetch_day(dt: date):
    url = BASE_URL.format(loc=LOCATION)
    ymd = dt.strftime("%Y%m%d")
    params = {"apiKey": API_KEY, "units": UNITS, "startDate": ymd, "endDate": ymd}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2)
            r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"[skip] {dt}: HTTP {r.status_code} {r.text[:100]}")
            return []

    data = r.json()
    obs = data.get("observations", []) or []
    rows = []
    for o in obs:
        t_local = datetime.fromtimestamp(o["valid_time_gmt"], tz=timezone.utc).astimezone(TZ_LOCAL)

        # Windows-safe 12-hour format:
        time_12h = t_local.strftime("%I:%M %p").lstrip("0")

        rows.append({
            "Date": t_local.date().isoformat(),
            "Time": time_12h,
            "Temperature": o.get("temp"),
            "Condition": o.get("wx_phrase"),
        })
    return rows


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

all_rows = []
for d in daterange(START, END):
    all_rows.extend(fetch_day(d))
    time.sleep(0.3)


if not all_rows:
    print("No data collected. Check API key/location/dates.")
else:
    df = pd.DataFrame(all_rows)

    # order & sort
    cols = ["Date","Time","Temperature","Condition"]
    df = df.reindex(columns=cols)

    # sort by Date then clock time
    def to_24h(t):
        for fmt in ("%I:%M %p",):  # add "%#I:%M %p" for Windows if needed
            try:
                return datetime.strptime(t, fmt).time()
            except Exception:
                pass
        return None

    df["_t"] = df["Time"].apply(to_24h)
    df.sort_values(["Date","_t"], inplace=True, kind="mergesort")
    df.drop(columns="_t", inplace=True)

    out = f"izmir_weather_{START.isoformat()}_to_{END.isoformat()}_Gaziemir.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows → {out}")
