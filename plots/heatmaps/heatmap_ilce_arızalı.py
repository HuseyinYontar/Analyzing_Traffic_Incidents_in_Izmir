import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel(r"..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce.xlsx")

df_ariza = df[df["TUR"].isin(["Arızalı"])].copy()


df_ariza["KAZA_ZAMANI"] = pd.to_datetime(
    df_ariza["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce"
)
df_ariza["HOUR"] = df_ariza["KAZA_ZAMANI"].dt.hour


df_ariza = df_ariza.dropna(subset=["ILCE", "HOUR"])


heatmap_data = df_ariza.groupby(["ILCE", "HOUR"]).size().unstack(fill_value=0)


plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor="gray"
)
plt.title("Heatmap of 'Arızalı' Accidents by District (İlçe) and Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("District (İlçe)")
plt.tight_layout()
plt.show()
