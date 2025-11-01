from path_getter import get_path_for_plotting
import pandas as pd
import matplotlib.pyplot as plt



file_path = get_path_for_plotting()
df = pd.read_excel(file_path)

df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"])
df["SAAT"] = df["KAZA_ZAMANI"].dt.hour
top5_ilce = df["ILCE"].value_counts().head(5).index.tolist()

df_top5 = df[df["ILCE"].isin(top5_ilce)]
hourly_counts=(
    df_top5
    .groupby(["ILCE","SAAT"])
    .size()
    .reset_index(name="KAZA_SAYISI")
)

plt.figure(figsize=(10, 6))

for ilce in top5_ilce:
    sub = hourly_counts[hourly_counts["ILCE"] == ilce]
    plt.plot(sub["SAAT"], sub["KAZA_SAYISI"], marker="o", label=ilce)

plt.title("Top 5 İlçe – Hourly Accident Distribution", fontsize=14)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)
plt.xticks(range(0, 24))
plt.legend(title="İlçe", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()