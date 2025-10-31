from path_getter import get_path_for_plotting
import pandas as pd


df = pd.read_excel(get_path_for_plotting())

df["STREET_AND_LOCATION"] = df["CADDE"] +" "+ df["KONUM"]

total_occurrence = (
    df.groupby(["STREET_AND_LOCATION"]).size().sort_values(ascending=False)
)

print(total_occurrence.to_string())

total_occurrence.to_excel("hot_spots_street_plus_location.xlsx")