import pandas as pd

# Load your CSV file
file_path = "../yandex_izmir_allrows_part2only_filled.csv"
df = pd.read_csv(file_path)

# === Option 1: If you know the column name ===
column_name = "y_first_addr_part2"  # change this to your actual column name

# Find rows with missing or empty values in that column
no_result_rows = df[df[column_name].isna() | (df[column_name].astype(str).str.strip() == '')]

print(f"Number of rows with no result in '{column_name}':", len(no_result_rows))
print("\nSample of rows with no result:")
print(no_result_rows.head())
print(no_result_rows.shape)

# Optionally save these rows to a new CSV
no_result_rows.to_csv("rows_with_no_result.csv", index=False)




#Karşıyaka
#Konak
#
#Konak
#



