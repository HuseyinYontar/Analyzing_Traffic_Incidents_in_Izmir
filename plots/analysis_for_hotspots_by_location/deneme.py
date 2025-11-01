import folium
import pandas as pd

# Example: streets with coordinates
data = {
    "street": ["Akçay Caddesi", "Gaziler Caddesi", "Altınyol", "Mustafa Kemal Bulvarı"],
    "lat": [38.333, 38.423, 38.457, 38.412],
    "lon": [27.139, 27.140, 27.120, 27.128],
    "count": [120, 98, 87, 65]
}
df = pd.DataFrame(data)

# Create a base map centered around Izmir
m = folium.Map(location=[38.42, 27.14], zoom_start=12)

# Add street hotspots as circle markers
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=row["count"] / 5,  # size proportional to incidents
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.6,
        tooltip=f"{row['street']}: {row['count']} incidents"
    ).add_to(m)

m.save("izmir_hotspots.html")
