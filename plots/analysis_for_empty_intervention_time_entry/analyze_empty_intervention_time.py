from path_getter import get_path_for_plotting
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_excel(get_path_for_plotting())

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

missing_mudahale = df[df["MUDAHALE_ZAMANI"].isna()]


missing_per_day = (
    missing_mudahale.groupby(df["TARIH"].dt.date)
    .size().sort_index()
)

total_per_day = (
    df.groupby(df["TARIH"].dt.date)
    .size()
    .sort_index()
)

which_day_type = (
    df["CALISMA_DURUMU"].groupby(df["TARIH"].dt.date)
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
)

combined = pd.DataFrame({
    "EMPTY_MUDAHALE_COUNT": missing_per_day,
    "TOTAL_INCIDENTS": total_per_day,
    "DAY_TYPE": which_day_type
}).fillna({"EMPTY_MUDAHALE_COUNT": 0}).astype({"EMPTY_MUDAHALE_COUNT": int, "TOTAL_INCIDENTS": int})

combined = combined[combined["EMPTY_MUDAHALE_COUNT"] > 0].reset_index()
combined.columns = ["TARIH", "TOTAL_EMPTY_INTERVENTION", "TOTAL_INCIDENTS_THAT_DAY", "DAY_TYPE"]

#combined.to_excel("every_missing_intervention_entry_groupbyday.xlsx",index=False)
print(combined.to_string(index=False))
print("total entry:", len(combined))



missing_per_day.index = pd.to_datetime(missing_per_day.index)

full_range = pd.date_range(missing_per_day.index.min(), missing_per_day.index.max(), freq="D")
missing_daily = (
    missing_per_day.reindex(full_range, fill_value=0)
    .rename("EMPTY_MUDAHALE_COUNT")
)
missing_daily.index.name = "TARIH"

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(missing_daily.index, missing_daily.values, width=1.0)

ax.set_title("Daily Count of Missing MUDAHALE_ZAMANI")
ax.set_xlabel("Date")
ax.set_ylabel("Missing Count")


locator = mdates.AutoDateLocator(minticks=8, maxticks=12)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

#plt.savefig("missing_mudahale_daily.png", dpi=300, bbox_inches="tight")
plt.savefig("missing_intervention_entry_daily.pdf", bbox_inches="tight")

plt.show()

