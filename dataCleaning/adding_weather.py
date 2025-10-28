import pandas as pd


acc = pd.read_excel("accidents_çiğli_with_weather_hour.xlsx")
wx = pd.read_csv("izmir_weather_2021-12-01_to_2025-09-28_Cigli.csv")


acc["TARIH"] = pd.to_datetime(acc["TARIH"], errors="coerce").dt.date
acc["HAVA_DURUMU_OLCUM_ZAMANI"] = pd.to_datetime(acc["HAVA_DURUMU_OLCUM_ZAMANI"], errors="coerce").dt.strftime("%H:%M")


wx["Date"] = pd.to_datetime(wx["Date"], errors="coerce").dt.date
wx["Time"] = pd.to_datetime(wx["Time"], format="%I:%M %p", errors="coerce").dt.strftime("%H:%M")


acc["MERGE_KEY"] = acc["TARIH"].astype(str) + " " + acc["HAVA_DURUMU_OLCUM_ZAMANI"]
wx["MERGE_KEY"] = wx["Date"].astype(str) + " " + wx["Time"]


merged = pd.merge(acc, wx, on="MERGE_KEY", how="left", suffixes=("_ACC", "_WX"))


unmatched = merged[merged["Temperature"].isna()][["TARIH", "HAVA_DURUMU_OLCUM_ZAMANI"]]

print(f"Total accidents: {len(acc)}")
print(f"Matched: {len(merged) - len(unmatched)}")
print(f"Unmatched: {len(unmatched)}")

merged.to_excel("merged_accidents_weather.xlsx", index=False)
unmatched.to_excel("unmatched_accidents.xlsx", index=False)
print("✅ Merge completed and saved.")
