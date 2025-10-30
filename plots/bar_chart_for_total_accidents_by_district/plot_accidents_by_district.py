import  pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting
in_path = get_path_for_plotting()
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
