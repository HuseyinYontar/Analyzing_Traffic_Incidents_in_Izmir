import  pandas as pd
import matplotlib.pyplot as plt

in_path = r"..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce.xlsx"
df = pd.read_excel(in_path)



accident_counts = df["ILCE"].value_counts().sort_values(ascending=False)


plt.figure(figsize=(12,6))
accident_counts.plot(kind="bar", color="steelblue")
plt.title("Number of Accidents by District (İlçe)")
plt.xlabel("District (İlçe)")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
