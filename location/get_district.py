# yandex_geo_allrows_part2only.py
import json
import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# === CONFIG ===
INPUT_XLSX  = r"izbb-kaza-ariza-verileri-cleaned-direction-type-street.xlsx"
OUTPUT_XLSX = "yandex_izmir_allrows_part2only.xlsx"
OUTPUT_CSV  = "yandex_izmir_allrows_part2only.csv"
CITY_PATH   = "11505/izmir"
LL          = "27.137574%2C38.326442"
Z           = "13"
DELAY_SEC   = 0.5          # increase if you hit rate limits
TIMEOUT_SEC = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) Gecko/20100101 Firefox/141.0",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}

# --- Helpers ---
def find_col_case_insensitive(df, name):
    for col in df.columns:
        if col.lower() == name.lower():
            return col
    return None

def build_search_url(query: str):
    q = quote(query, safe="")
    return f"https://yandex.com.tr/maps/{CITY_PATH}/search/{q}/?ll={LL}&z={Z}"

def fetch_final_url(session, query: str):
    url = build_search_url(query)
    r = session.get(url, headers=HEADERS, timeout=TIMEOUT_SEC, allow_redirects=True)
    r.raise_for_status()
    return r.url

def parse_state_view(html: str):
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/json", "class": "state-view"})
    if not scripts:
        return {}
    raw = (scripts[-1].string or scripts[-1].text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}

def deep_find(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from deep_find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from deep_find(v, key)

def pick_first_result_fields(state):
    for items in deep_find(state, "items"):
        if isinstance(items, list) and items:
            it = items[0]
            name = it.get("name") or it.get("title")
            address = it.get("address") or it.get("subtitle") or it.get("description")
            return name, address
    return None, None

def get_second_address_part(address: str):
    if not address:
        return ""
    parts = [p.strip() for p in address.split(",") if p.strip()]
    return parts[1] if len(parts) > 1 else ""

# --- Main ---
df = pd.read_excel(INPUT_XLSX)
col_cadde = find_col_case_insensitive(df, "Cadde")
col_konum = find_col_case_insensitive(df, "KONUm")
if not col_cadde or not col_konum:
    raise ValueError(f"Could not find 'Cadde' and 'KONUm' columns. Found: {list(df.columns)}")

df = df.copy()
df["y_query"] = (
    df[col_cadde].fillna("").astype(str).str.strip() + " " +
    df[col_konum].fillna("").astype(str).str.strip()
).str.strip()

for col in ["y_first_name", "y_first_address", "y_first_addr_part2", "y_status"]:
    df[col] = ""

with requests.Session() as s:
    total = len(df)
    for i, row in df.iterrows():
        query = row["y_query"]
        if not query:
            df.at[i, "y_status"] = "EMPTY QUERY"
            continue

        try:
            if (i + 1) % 25 == 0 or i == 0:
                print(f"[{i+1}/{total}] Searching: {query}")
            final_url = fetch_final_url(s, query)
            resp = s.get(final_url, headers=HEADERS, timeout=TIMEOUT_SEC)
            resp.raise_for_status()

            state = parse_state_view(resp.text)
            name, address = pick_first_result_fields(state)

            df.at[i, "y_first_name"] = name or ""
            df.at[i, "y_first_address"] = address or ""
            df.at[i, "y_first_addr_part2"] = get_second_address_part(address)
            df.at[i, "y_status"] = "OK" if (name or address) else "No result"

        except Exception as e:
            df.at[i, "y_status"] = f"ERROR: {e}"

        time.sleep(DELAY_SEC)

# --- Save ---
df.to_excel(OUTPUT_XLSX, index=False)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Done! Processed ALL rows.\nSaved to:\n- {OUTPUT_XLSX}\n- {OUTPUT_CSV}")
