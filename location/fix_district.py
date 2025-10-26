import re
import pandas as pd

# === File paths ===
in_path  = "../yandex_izmir_allrows_part2only_filled.csv"
out_path = "../yandex_izmir_allrows_part2only_filled_updated.csv"
not_found_path = "../not_found_ilce.csv"

# === Load data ===
df = pd.read_csv(in_path)

# === Columns ===
SOURCE_COL = "y_first_address"
TARGET_COL = "y_first_addr_part2"
QUERY_COL = "y_query"

if TARGET_COL not in df.columns:
    df[TARGET_COL] = ""

# === Valid İzmir districts ===
all_ilces = [
    "Aliağa", "Balçova", "Bayındır", "Bayraklı", "Bergama", "Beydağ", "Bornova", "Buca",
    "Çeşme", "Çiğli", "Dikili", "Foça", "Gaziemir", "Güzelbahçe", "Karabağlar", "Karaburun",
    "Karşıyaka", "Kemalpaşa", "Kınık", "Kiraz", "Konak", "Menderes", "Menemen", "Narlıdere",
    "Ödemiş", "Seferihisar", "Selçuk", "Tire", "Torbalı", "Urla"
]

# === Helpers ===
def norm(s: str) -> str:
    """Unicode-friendly normalization for case-insensitive matching."""
    if not isinstance(s, str):
        return ""
    return s.casefold().strip()

ilces_norm = [norm(x) for x in all_ilces]
ilces_set = set(ilces_norm)
patterns = [(name, re.compile(rf"\b{re.escape(norm(name))}\b", flags=re.UNICODE)) for name in all_ilces]

def choose_ilce_from_address(address: str):
    """Try to find a known ilçe name in the address string."""
    a = norm(address)
    if not a:
        return None
    for original, pat in patterns:
        if pat.search(a):
            return original
    for original in all_ilces:
        if norm(original) in a:
            return original
    return None

# === Track before update ===
before = df[TARGET_COL].astype(str).copy()

# === Main update ===
def update_row(row):
    current = str(row.get(TARGET_COL, ""))
    current_norm = norm(current)
    if current_norm in ilces_set:
        return current
    full_addr = row.get(SOURCE_COL, "")
    inferred = choose_ilce_from_address(full_addr)
    return inferred if inferred else current

df[TARGET_COL] = df.apply(update_row, axis=1)

# === Identify changed and not found rows ===
changed_mask = before.ne(df[TARGET_COL])
still_not_found_mask = ~df[TARGET_COL].astype(str).map(norm).isin(ilces_set)
not_found_df = df[still_not_found_mask].copy()

num_changed = int(changed_mask.sum())
num_not_found = len(not_found_df)

# === Print summary ===
print("\n────────── SUMMARY ──────────")
print(f"✅ Updated rows: {num_changed}")
print(f"⚠️  Rows still without valid ilçe: {num_not_found}")
print(f"📊 Total rows in dataset: {len(df)}")
print(f"📈 Coverage: {100 * (len(df) - num_not_found) / len(df):.2f}% have valid ilçes")

# === Show example updates ===
if num_changed:
    sample_changes = df.loc[changed_mask, [SOURCE_COL]].copy()
    sample_changes["old"] = before[changed_mask].values
    sample_changes["new"] = df.loc[changed_mask, TARGET_COL].values
    with pd.option_context("display.max_colwidth", 150):
        print("\n🔄 Example updated rows:")
        print(sample_changes.head(10))

# === İlçe frequency summary ===
print("\n🏙️  İlçe frequency after update:")
print(df[TARGET_COL].value_counts().head(20))

# === Print not found details ===
if num_not_found:
    print("\n❌ Example 'not found' rows:")
    with pd.option_context("display.max_colwidth", 150):
        print(not_found_df[[SOURCE_COL, TARGET_COL]].head(10))

    print("\n🧾 Full list of NOT FOUND addresses:")
    for i, addr in enumerate(not_found_df[SOURCE_COL].tolist(), start=1):
        print(f"{i:03d}. {addr}")

    # === Unique y_queries from not found ===
    if QUERY_COL in not_found_df.columns:
        print("\n🔍 Unique y_queries from NOT FOUND rows (with frequencies):")
        query_counts = not_found_df[QUERY_COL].value_counts(dropna=False)
        print(query_counts.to_string())
    else:
        print("\n⚠️  Column 'y_query' not found in dataset, skipping frequency summary.")

# === Save outputs ===
df.to_csv(out_path, index=False)
not_found_df.to_csv(not_found_path, index=False)

print(f"\n💾 Saved updated file to: {out_path}")
print(f"💾 Saved not-found rows to: {not_found_path}")


# === (SECOND-PASS) y_query -> ilçe mapping ===
# Paste your big mapping dict below
add_ilce = {
  "Akçay Caddesi 6. Sanayi Kavşak İçi": "Gaziemir",
  "Akçay Caddesi 6. Sanayi Kavşağı": "Gaziemir",
  "Akçay Caddesi 6.Sanayi Kavşağı": "Gaziemir",
  "Akçay Caddesi Aktepe Ayrımı Öncesi": "Gaziemir",
  "Akçay Caddesi Aktepe Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Aktepe Üst Geçit Altı": "Gaziemir",
  "Akçay Caddesi Alt Geçit": "Gaziemir",
  "Akçay Caddesi Gaziemir Alt Geçit İçi": "Gaziemir",
  "Akçay Caddesi Gaziemir Alt geçit İçi": "Gaziemir",
  "Akçay Caddesi Gaziemir Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Gaziemir alt Geçit İçi": "Gaziemir",
  "Akçay Caddesi Gediz Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Havalimanı Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Kipa Alt Geçit İçi": "Gaziemir",
  "Akçay Caddesi Orcaner Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Sarnıç Alt Geçit Girişi": "Gaziemir",
  "Akçay Caddesi Sarnıç Alt Geçit Çıkışı": "Gaziemir",
  "Akçay Caddesi Sarnıç Alt Geçit İ": "Gaziemir",
  "Akçay Caddesi Sarnıç Altgeçit Girişi": "Gaziemir",
  "Akçay Caddesi Sarnıç Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Sarnıç Kavşağı Üst Geçit Çıkışı": "Gaziemir",
  "Akçay Caddesi Sarnıç Kavşağı İçi": "Gaziemir",
  "Akçay Caddesi Sarnıç Kavşk İçi": "Gaziemir",
  "Akçay Caddesi Sarnıç Köprü Girişi": "Gaziemir",
  "Akçay Caddesi Sarnıç Köprü Çıkışı": "Gaziemir",
  "Akçay Caddesi Sarnıç Köprü Üstü": "Gaziemir",
  "Akçay Caddesi Sarnıç köprü Çıkışı": "Gaziemir",
  "Akçay Caddesi Serbest Bölge Alt Geçit Üstü": "Gaziemir",
  "Akçay Caddesi Yaşayanlar Kavşağı": "Konak",
  "Akçay Caddesi Çevreyolu Katılım Öncesi": "Gaziemir",
  "Akçay Caddesi Çevreyolu Kavşak İçi": "Gaziemir",
  "Akçay Caddesi Çevreyolu Kavşağı": "Gaziemir",
  "Akçay Caddesi Çevreyolu Kavşağı Öncesi": "Gaziemir",
  "Akçay Caddesi Çevreyolu Köprü Altı": "Gaziemir",
  "Akçay Caddesi Çevreyolu Köprü Sonrası": "Gaziemir",
  "Akçay Caddesi Çevreyolu Viyadük Altı": "Gaziemir",
  "Alim Dağhan Caddesi Şemikler Kavşağı Öncesi": "Karşıyaka",
  "Altınyol Caddesi Adnan Kahveci Köprü Altı": "Bayraklı",
  "Altınyol Caddesi Adnan Kahveci Köprü altı": "Bayraklı",
  "Altınyol Caddesi Adnan Kahveci Köprü Üstü": "Bayraklı",
  "Altınyol Caddesi Adnan Kahveci Köptü Altı": "Bayraklı",
  "Altınyol Caddesi Alsancak Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Alsancak Yol Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Ankara Caddesi Yol Ayrımı": "Bayraklı",
  "Altınyol Caddesi Bayraklı Belediyesi Sonrası": "Bayraklı",
  "Altınyol Caddesi Bayraklı Köprü Altı": "Bayraklı",
  "Altınyol Caddesi DGM Köprü Üstü": "Konak",
  "Altınyol Caddesi DGM Meles Katılımı": "Konak",
  "Altınyol Caddesi Liman Ayrımı Sonrası": "Konak",
  "Altınyol Caddesi Liman Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Liman Caddesi Girişi": "Bayraklı",
  "Altınyol Caddesi Liman Yol Ayrımı Sonrası": "Konak",
  "Altınyol Caddesi Liman Yol Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Liman Yol ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Meles": "Konak",
  "Altınyol Caddesi Meles Alsancak Ayrım": "Konak",
  "Altınyol Caddesi Meles Alsancak Ayrımı": "Konak",
  "Altınyol Caddesi Meles Alsancak Ayrımı Sonrası": "Konak",
  "Altınyol Caddesi Meles Alsancak Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Meles Alsancak Basmane Ayrımı": "Konak",
  "Altınyol Caddesi Meles Alsancak Yol Ayrımı": "Konak",
  "Altınyol Caddesi Meles Alsancak Yol ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Meles Alsancak ve DGM Ayrımı": "Konak",
  "Altınyol Caddesi Meles Altınyol Katılımı": "Konak",
  "Altınyol Caddesi Meles Altınyol Katılımı Sonrası": "Konak",
  "Altınyol Caddesi Meles DGM Katılım": "Konak",
  "Altınyol Caddesi Meles DGM Katılım Öncesi": "Konak",
  "Altınyol Caddesi Meles DGM Katılımı": "Konak",
  "Altınyol Caddesi Meles DGM Köprü Girişi": "Bayraklı",
  "Altınyol Caddesi Meles DGM Sonrası": "Konak",
  "Altınyol Caddesi Meles Delta Hizası": "Bayraklı",
  "Altınyol Caddesi Meles Deltası Hizası": "Bayraklı",
  "Altınyol Caddesi Meles Halkapınar Ayrımı": "Konak",
  "Altınyol Caddesi Meles Halkapınar ayrımı sonrası": "Konak",
  "Altınyol Caddesi Meles Katılım Öncesi": "Konak",
  "Altınyol Caddesi Meles Katılımı Öncesi": "Konak",
  "Altınyol Caddesi Meles Liman Ayrımı": "Konak",
  "Altınyol Caddesi Meles Liman Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Meles Yol Ayrımı Sonrası": "Konak",
  "Altınyol Caddesi Meles Yol Ayrımı Öncesi": "Konak",
  "Altınyol Caddesi Meles Zafer Payzın Katılımı": "Konak",
  "Altınyol Caddesi Meles Zafer Payzın Katılımı Sonrası": "Konak",
  "Altınyol Caddesi Turan Üst Geçit Altı": "Bayraklı",
  "Altınyol Caddesi Zafer Payzın Köprü Altı": "Konak",
  "Altınyol Caddesi Zafer Payzın Köprü Girişi": "Konak",
  "Altınyol Caddesi İlave Şerit Girişi": "Konak",
  "Altınyol Caddesi İlave Şerit Öncesi": "Bayraklı",
  "Anadolu Caddesi": "Bayraklı",
  "Anadolu Caddesi 10 Nisan Köprü Altı": "Bayraklı",
  "Anadolu Caddesi 10 Nisan Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi Adliye Üst Geçit Öncesi": "Bayraklı",
  "Anadolu Caddesi Adnan Kahveci Köprü": "Bayraklı",
  "Anadolu Caddesi Adnan Kahveci Köprü Sonrası": "Bayraklı",
  "Anadolu Caddesi Adnan Kahveci Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi Adnan Kahveci Köprüsü": "Bayraklı",
  "Anadolu Caddesi Alaybey Ayrımı": "Karşıyaka",
  "Anadolu Caddesi Alaybey Çıkışı": "Karşıyaka",
  "Anadolu Caddesi Ayyıldız Kavşak İçi": "Karşıyaka",
  "Anadolu Caddesi Ayyıldız Kavşağı": "Karşıyaka",
  "Anadolu Caddesi Ayyıldız Kavşağı Öncesi": "Karşıyaka",
  "Anadolu Caddesi Baraklı Belediyesi Hizası": "Bayraklı",
  "Anadolu Caddesi Bayraklı Belediye Hizası": "Bayraklı",
  "Anadolu Caddesi Bayraklı Belediyesi Hizası": "Bayraklı",
  "Anadolu Caddesi Bayraklı Belediyesi Sonrası": "Bayraklı",
  "Anadolu Caddesi Bayraklı Belediyesi Öncesi": "Bayraklı",
  "Anadolu Caddesi Bayraklı Üst Geçit": "Bayraklı",
  "Anadolu Caddesi Benzinlik Öncesi": "Bayraklı",
  "Anadolu Caddesi Bornova-Karşıyaka Yol Ayrımı": "Bayraklı",
  "Anadolu Caddesi Bostanlı İtfaiye Hizası": "Çiğli",
  "Anadolu Caddesi Bostanlı İtfaiye Kavşak İçi": "Çiğli",
  "Anadolu Caddesi Bostanlı İtfaiye Sonrası": "Çiğli",
  "Anadolu Caddesi Bostanlı İtfaiye Öncesi": "Çiğli",
  "Anadolu Caddesi Cumhuriyet Çıkışı": "Karşıyaka",
  "Anadolu Caddesi DGM": "Bayraklı",
  "Anadolu Caddesi DGM Köprü Girişi": "Bayraklı",
  "Anadolu Caddesi DGM Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi DGM Köprü İnişi": "Bayraklı",
  "Anadolu Caddesi Dedebaşı Kavşak İçi": "Karşıyaka",
  "Anadolu Caddesi Dedebaşı Kavşağı Öncesi": "Karşıyaka",
  "Anadolu Caddesi Ege Deniz Bölge Komutanlığı Önü": "Karşıyaka",
  "Anadolu Caddesi Ege Üniversitesi Karşıyaka Yerleşkesi": "Karşıyaka",
  "Anadolu Caddesi Ege Üniversitesi Karşıyaka Yerleşkesi Hizası": "Karşıyaka",
  "Anadolu Caddesi Evka 5 Kavşağı Sonrası": "Çiğli",
  "Anadolu Caddesi Evka 5 Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Evka 5 Köprülü Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Evka 5 Öncesi": "Çiğli",
  "Anadolu Caddesi Ferdi Varol Yaya Üst Geçit Altı": "Bayraklı",
  "Anadolu Caddesi Gaziemir Alt Geçit İçi": "Gaziemir",
  "Anadolu Caddesi Girne Bulvarı": "Karşıyaka",
  "Anadolu Caddesi Girne Kavşak İçi": "Karşıyaka",
  "Anadolu Caddesi Girne Kavşağı Viyadükaltı": "Karşıyaka",
  "Anadolu Caddesi Girne Köprü Katılımı Sonrası": "Karşıyaka",
  "Anadolu Caddesi Girne Köprü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Girne Köprü Öncesi": "Karşıyaka",
  "Anadolu Caddesi Girne Köprüsü": "Karşıyaka",
  "Anadolu Caddesi Girne Köprüsü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Girne Köprüsü Öncesi": "Karşıyaka",
  "Anadolu Caddesi Harmandalı Alt Geçit Öncesi": "Çiğli",
  "Anadolu Caddesi Harmandalı Alt Geçit İçi": "Çiğli",
  "Anadolu Caddesi Harmandalı Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Karya Evleri Hizası": "Çiğli",
  "Anadolu Caddesi Katlı Pazaryeri": "Bayrakli",
  "Anadolu Caddesi Katlı Pazaryeri Hizası": "Bayraklı",
  "Anadolu Caddesi Katlı Pazaryeri Karşısı": "Bayraklı",
  "Anadolu Caddesi Katlı Pazaryeri Öncesi": "Bayraklı",
  "Anadolu Caddesi Maltep Kavşağı": "Çiğli",
  "Anadolu Caddesi Maltepe Kavşak İçi": "Çiğli",
  "Anadolu Caddesi Maltepe Kavşağı": "Çiğli",
  "Anadolu Caddesi Maltepe Kavşağı Sonrası": "Çiğli",
  "Anadolu Caddesi Maltepe Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Meles Deltası": "Konak",
  "Anadolu Caddesi Naldöken Köprsü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü Hizası": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü SÖncesi": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü sonrası": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü Öncesi": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü Üncesi": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprü Üzeri": "Karşıyaka",
  "Anadolu Caddesi Naldöken Köprüsü Öncesi": "Karşıyaka",
  "Anadolu Caddesi Naldöken Yaya Geçidi Sonrası": "Karşıyaka",
  "Anadolu Caddesi Naldöken köprü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Sarnıç Alt Geçit İçi": "Gaziemir",
  "Anadolu Caddesi Serinkuyu Kavşak İçi": "Bayraklı",
  "Anadolu Caddesi Shell Benzinlik Hizası": "Bayraklı",
  "Anadolu Caddesi Smyrna Yol Ayrımı": "Konak",
  "Anadolu Caddesi Soğukkuyu Benzinlik Önü": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşak İçi": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı Hizası": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı Sonrası": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı öncesi": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Üst Geçit Altı": "Bayraklı",
  "Anadolu Caddesi Tepecik Hastanesi Karşıyaka Semt Pol. Hizası": "Karşıyaka",
  "Anadolu Caddesi Tersane Ayrımı": "Bayraklı",
  "Anadolu Caddesi Turan": "Bayraklı",
  "Anadolu Caddesi Turan Bayraklı Çıkışı": "Bayraklı",
  "Anadolu Caddesi Turan Köprü": "Bayraklı",
  "Anadolu Caddesi Turan Köprü Üzeri": "Bayraklı",
  "Anadolu Caddesi Turan Köprüsü": "Bayraklı",
  "Anadolu Caddesi Turan Otobüs Durakları": "Bayraklı",
  "Anadolu Caddesi Turan ÜSt Geçit Öncesi": "Bayraklı",
  "Anadolu Caddesi Turan Üst Geçit": "Bayraklı",
  "Anadolu Caddesi Turan Üst Geçit Altı": "Bayraklı",
  "Anadolu Caddesi Turan Üst Geçit Öncesi": "Bayraklı",
  "Anadolu Caddesi Turan Üst geçit Öncesi": "Bayraklı",
  "Anadolu Caddesi Turan Üstgeçit": "Bayraklı",
  "Anadolu Caddesi Yeni Girne Kavşak İçi": "Karşıyaka",
  "Anadolu Caddesi Yeni Girne Köprü Sonrası": "Karşıyaka",
  "Anadolu Caddesi Zafer Payzın Kavşak İçi": "Konak",
  "Anadolu Caddesi Zafer Payzın Köprü Öncesi": "Konak",
  "Anadolu Caddesi Ziya Gökalp Kültür Merkezi Öncesi": "Konak",
  "Anadolu Caddesi Çevreyolu Girne Bulvarı": "Karşıyaka",
  "Anadolu Caddesi Çiğli Alt Geçit": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit Hizası": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit İçi": "Çiğli",
  "Anadolu Caddesi Çiğli Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli Maltepe Işıkları": "Çiğli",
  "Anadolu Caddesi Çiğli Vergi Dairesi Hizası": "Çiğli",
  "Anadolu Caddesi Çiğli Vergi Dairesi Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli İtfaiye Sonrası": "Çiğli",
  "Anadolu Caddesi Çiğli İtfaiye Önü": "Çiğli",
  "Anadolu Caddesi İlave Şerit": "Konak",
  "Anadolu Caddesi İlave Şerit Çıkışı": "Konak",
  "Anadolu Caddesi Şemikler Kavşak İçi": "Karşıyaka",
  "Anadolu Caddesi Şemikler Kavşağı": "Karşıyaka",
  "Anadolu Caddesi Şemikler Kavşağı Sonrası": "Karşıyaka",
  "Anadolu Caddesi Şemikler Kavşağı Öncesi": "Karşıyaka",
  "Anadolu Caddesi Şemikler Köprü Öncesi": "Karşıyaka",
  "Anadolu Caddesi Şoğukkuyu Kavşak İçi": "Bayraklı",
  "Ankara Caddesi DGM Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Egemak Köorü Üstü": "Bornova",
  "Ankara Caddesi Egemak Köprü Altı": "Bornova",
  "Ankara Caddesi Egemak Köprü Girişi": "Bornova",
  "Ankara Caddesi Egemak Köprü Çıkışı": "Bornova",
  "Ankara Caddesi Egemak Köprü Üstü": "Bornova",
  "Ankara Caddesi Fatih Caddesine Dönüş": "Bornova",
  "Ankara Caddesi Hilal Köprü Üstü": "Konak",
  "Ankara Caddesi Mahvel Kavşağı Hizası": "Bornova",
  "Ankara Caddesi Mahvel Kavşağı Sonrası": "Bornova",
  "Ankara Caddesi Mahvel Kavşağı Öncesi": "Bornova",
  "Ankara Caddesi Manisa Kavşak İçi": "Bornova",
  "Ankara Caddesi Meles Katılımı Hizası": "Konak",
  "Ankara Caddesi Naldöken Kavşağı Viyadük Üstü": "Bornova",
  "Ankara Caddesi Nilüfer Kavşağı Viyadük Üstü": "Bornova",
  "Ankara Caddesi Nilüfer Kavşağı Öncesi": "Bornova",
  "Ankara Caddesi Osman Kibar Alt Geçit Altı": "Bornova",
  "Ankara Caddesi Osman Kibar Kavşağı Sonrası": "Bornova",
  "Ankara Caddesi Osman Kibar Kavşağı Öncesi": "Bornova",
  "Ankara Caddesi Osman Kibar Tünel Çıkışı": "Bornova",
  "Ankara Caddesi Osman Kibar Tüneli Çıkışı": "Bornova",
  "Ankara Caddesi Semt Garajı Hizası": "Gaziemir",
  "Ankara Caddesi Turan Köprü Hizası": "Bayraklı",
  "Ankara Caddesi Turan Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Yağ Fabrikası Hizası": "Bornova",
  "Ankara Caddesi Yağ Fabrikası Öncesi": "Bornova",
  "Ankara Caddesi Zafer Payzın Köprü Girişi": "Konak",
  "Ankara Caddesi Zafer Payzın Köptü Üstü": "Konak",
  "Ankara Caddesi Zafer Payzın Meles Katılımı Öncesi": "Konak",
  "Ankara Caddesi Çevreyeyolu Ayrımı": "Bornova",
  "Ankara Caddesi Çevreyolu Ayrımı": "Bornova",
  "Ankara Caddesi Çevreyolu Bağlantı": "Bornova",
  "Ankara Caddesi Çevreyolu Girişi": "Bornova",
  "Ankara Caddesi Çevreyolu Katılı": "Bornova",
  "Ankara Caddesi Çevreyolu Katılımı": "Bornova",
  "Ankara Caddesi Çevreyolu Köprü Altı": "Bornova",
  "Ankara Caddesi Çevreyolu Köprüsü Sonrası": "Bornova",
  "Ankara Caddesi Çevreyolu Çıkışı": "Bornova",
  "Ankara Caddesi Çevreyolu İnişi": "Bornova",
  "Ankara Caddesi Özkanlar Köprü Üstü": "Bornova",
  "Atatürk Caddesi Cennet Vadisi Hizası": "Konak",
  "Atatürk Caddesi Gar Kavşağı": "Konak",
  "Atatürk Caddesi Hal Kavşak İçi": "Buca",
  "Atatürk Caddesi Sebze Hali Kavşak İçi": "Buca",
  "Aydın Hatboyu Caddesi Namık Kemal Caddesi Kesişimi": "Buca",
  "Aydınlar Caddesi Işıkkent Çevreyolu Viyadük Altı": "Bornova",
  "Basmane Meydan Meydan": "Konak",
  "Caher Dudayev Bulvarı AVM Önü": "Karşıyaka",
  "Caher Dudayev Bulvarı Otoyol Gelişi": "Karşıyaka",
  "Caher Dudayev Bulvarı Otoyol Girişi": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu Ayrımı": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu Bağlantı Sonrası": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu İnişi": "Karşıyaka",
  "Cemal Gürsel Caddesi Alaybey Tersane Hizası": "Karşıyaka",
  "Cemal Gürsel Caddesi Anıt Hizası": "Karşıyaka",
  "Cemal Gürsel Caddesi Atilla İlhan Anıtı Hizası": "Karşıyaka",
  "Cemal Gürsel Caddesi Bostanlı İskele Öncesi": "Karşıyaka",
  "Cemal Gürsel Caddesi Bostanlı İskelesi Öncesi": "Karşıyaka",
  "Cemal Gürsel Caddesi Muammer Aksoy Parkı Hizası": "Karşıyaka",
  "Cemal Gürsel Caddesi Naldöken Köprü Üstü": "Bayraklı",
  "Cemal Gürsel Caddesi Naldöken Köprü İnişi": "Bayraklı",
  "Cemal Gürsel Caddesi Yelken Kulubü Hizası": "Karşıyaka",
  "Cemal Gürsel Caddesi İskele Hizası": "Karşıyaka",
  "Cengizhan Caddesi Ege Üniversitesi Konukevi Hizası": "Bornova",
  "Cumhuriyet Bulvarı Borsa Kavşak İçi": "Konak",
  "Cumhuriyet Bulvarı Borsa Kavşağı": "Konak",
  "Cumhuriyet Bulvarı Cumhuriyet Meydanı Kavşağı": "Konak",
  "Cumhuriyet Bulvarı Fevzipaşa Bulvarı Girişi": "Konak",
  "Cumhuriyet Bulvarı Konak Pier Yaya Üst Geçit Altı": "Konak",
  "Doğuş Caddesi Tınavtepe Kavşağı": "Buca",
  "Doğuş Caddesi Tınaztepe Kampüs Girişi": "Buca",
  "Doğuş Caddesi Tınaztepe Kavşak İçi": "Buca",
  "Doğuş Caddesi Tınaztepe Kavşağı": "Buca",
  "Doğuş Caddesi Özel Hastane Kavşak İçi": "Buca",
  "Eski İzmir Caddesi Köstence Köprü Öncesi": "Karabağlar",
  "Eski İzmir Caddesi Köstence Köprü Üstü": "Karabağlar",
  "Eşrefpaşa Caddesi Agora Kavşağı Sonrası": "Konak",
  "Eşrefpaşa Caddesi Bayramyeri Kavşağı": "Konak",
  "Eşrefpaşa Caddesi Buca köprü Altı": "Konak",
  "Eşrefpaşa Caddesi Eski İzmir Caddesi Kesişimi": "Konak",
  "Eşrefpaşa Caddesi Kestelli Caddesi Girişi": "Konak",
  "Eşrefpaşa Caddesi Kılıcı Mescit Hizası": "Konak",
  "Fatih Caddesi 3.Sanayi Sitesi": "Bornova",
  "Fatih Caddesi Egemak Kavşak İçi": "Bornova",
  "Fatih Caddesi Egemak Kavşağı": "Bornova",
  "Fatih Caddesi Egemak Köprü Altı": "Bornova",
  "Fatih Caddesi Egemak Köprü altı": "Bornova",
  "Fatih Caddesi Stadyum Hizası": "Bornova",
  "Fatih Caddesi Vakıflar Kavşak İçi": "Bornova",
  "Fatih Caddesi Vakıflar Kavşağı": "Bornova",
  "Fevzi Çakmak Caddesi 161 Sokak Kesişimi": "Buca",
  "Fevzipaşa Bulvarı Basmana Gar Önü": "Konak",
  "Fevzipaşa Bulvarı Basmane Gar Önü": "Konak",
  "Fevzipaşa Caddesi Konak Üst Geçit Altı": "Konak",
  "Gazi Atatürk Bulvarı 73 Sokak Kesişimi": "Konak",
  "Gazi Bulvarı Borsa KAvşağı": "Konak",
  "Gazi Bulvarı Borsa Kavşak İçi": "Konak",
  "Gazi Bulvarı Borsa Kavşağı": "Konak",
  "Gazi Bulvarı Eski İtfaiye Kavşak İçi": "Konak",
  "Gazi Bulvarı Halit Ziya Bulvarı Kesişimi": "Konak",
  "Gazi Bulvarı İtfaiye Kavşak İçi": "Konak",
  "Gazi Bulvarı İtfaiye Kavşağı Sonrası": "Konak",
  "Gazi Bulvarı İtfaiye Kavşağı Öncesi": "Konak",
  "Gazi Bulvarı Şair Eşref Bulvarı Kesişimi": "Konak",
  "Gaziler Caddesi Boğaziçi Kavşak İçi": "Konak",
  "Gaziler Caddesi Kemer Alt Geçit Altı": "Konak",
  "Gaziler Caddesi Kemer Alt Geçit Girişi": "Konak",
  "Gaziler Caddesi Kemer Alt Geçit Çıkışı": "Konak",
  "Gaziler Caddesi Kemer Alt Geçit İ": "Konak",
  "Gaziler Caddesi Tepecik Köprü Altı": "Konak",
  "Gaziler Caddesi Tepecik Köprü Üstü": "Konak",
  "Gaziler Caddesi Tepecik Üstgeçit Altı": "Konak",
  "Gaziler Caddesi Yeşildere Viyadük Altı": "Konak",
  "Gaziosmanpaşa Bulvarı İtfaiye Kavşak İçi": "Konak",
  "Girne Bulvarı Girne Kavşak İçi": "Karşıyaka",
  "Girne Bulvarı Girne Kavşağı Öncesi": "Karşıyaka",
  "Girne Bulvarı Girne Köprü Sonrası": "Karşıyaka",
  "Girne Bulvarı Girne Köprü Üstü": "Karşıyaka",
  "Girne Bulvarı Girne Köprüsü": "Karşıyaka",
  "Girne Bulvarı Girne Kültür Park": "Karşıyaka",
  "Girne Bulvarı Girne Pazar Alanı": "Karşıyaka",
  "Girne Bulvarı Girne Viyadük Üstü": "Karşıyaka",
  "Girne Bulvarı Girne-Günsazak Bulvarı Kesişimi": "Karşıyaka",
  "Girne Bulvarı Karaksan Kavşak İçi": "Karşıyaka",
  "Girne Bulvarı Lunapark Kavşak İçi": "Karşıyaka",
  "Girne Bulvarı Ordu Bulvarı Kavşak İçi": "Karşıyaka",
  "Girne Bulvarı Soğukkuyu Kavşağı Öncesi": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Kavşak İçi": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Kavşağı": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Kavşağı Sonrası": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Köprü Sonrası": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Köprü Üstü": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Viyadük Öncesi": "Karşıyaka",
  "Girne Bulvarı Yeni Girne Viyadük Üstü": "Karşıyaka",
  "Girne Bulvarı Özel Hastane Kavşağı": "Karşıyaka",
  "Halide Edip Adıvar Caddesi Buca Katılımı Sonrası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü Sonrası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü Çıkışı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü Üstü": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprü İnişi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Köprüsü Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı Sonrası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Buca, Karabağlar Ayrımı Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi DGM Köprü Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Halide Edip Adıvar Caddesi Katılımı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Köstence Kavşağı Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Köstence Köprü Sonrası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Köstence Köprü Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Köstence Üst Geçit Sonrası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Yeşildere Caddesi'ne Katılım": "Karabağlar",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı Kavşak İçi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı SÖncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı Öncesi": "Karabağlar",
  "Halide Edip Adıvar Caddesi Üçyol Meydanı": "Karabağlar",
  "Halide Edip Adıvar Caddesi Şelale Hizası": "Karabağlar",
  "Halide Edip Adıvar Caddesi Şoförler ve Otomobilciler Esnaf Odası Hizası": "Karabağlar",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durakları Hizası": "Konak",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durakları Önü": "Konak",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durağı Hizası": "Konak",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durakları Hizası": "Konak",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durakları Öncesi": "Konak",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durağı Hizası": "Konak",
  "Halil Rıfat Paşa Caddesi Oyuncak Müzesi Öncesi": "Konak",
  "Halit Ziya Bulvarı Gazi Bulvarı Kesişimi": "Konak",
  "Hasan Ali Yücel Bulvarı Bostanlı Pazaryeri Öncesi": "Karşıyaka",
  "Hasan Ali Yücel Bulvarı Mavişehir Girişi": "Karşıyaka",
  "Haydar Aliyev Bulvarı Marina Kavşak Öncesi": "Balçova",
  "Haydar Aliyev Bulvarı Marina Kavşak İçi": "Balçova",
  "Haydar Aliyev Bulvarı Marina Kavşağı": "Balçova",
  "Haydar Aliyev Bulvarı Marina Kavşağı Öncesi": "Balçova",
  "Haydar Aliyev Bulvarı Marina Kavşağı İçinde": "Balçova",
  "Haydar Aliyev Caddesi Folkart Kavşak İçi": "Bayraklı",
  "Homeros Bulvarı Konak Tünel Girişi": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı Sonrası": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı Yeşildere Katılımı": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı Öncesi": "Konak",
  "Kamil Tunca Caddesi 1. Sanayi Kavşak İçi": "Bornova",
  "Kamil Tunca Caddesi Abdi İpekçi Caddesi Kesişimi": "Bornova",
  "Kamil Tunca Caddesi Vakıflar Kavşak İçi": "Bornova",
  "Kaynak Caddesi Aydın Hatboyu Köprü Altı": "Buca",
  "Konak Tüneli Konak Tüneli Girişi": "Konak",
  "Konak Tüneli T2": "Konak",
  "Konak Tüneli Tünel Girişi": "Konak",
  "Liman Caddesi Liman Köprü Üstü": "Konak",
  "Liman Caddesi Liman Viyadük Üstü": "Konak",
  "Manas Bulvarı Adliye Kavşak İçi": "Konak",
  "Mehmet Akif Ersoy Caddesi Beyazıt Aykut Parkı": "Buca",
  "Mehmet Akif Ersoy Caddesi Buca Köprü Altı": "Buca",
  "Mehmet Akif Ersoy Caddesi Buca köprü Altı": "Buca",
  "Mehmet Akif Ersoy Caddesi Kızılçullu Su Kemerleri Hizası": "Buca",
  "Mehmet Akif Ersoy Caddesi Nato Kavşak İçi": "Buca",
  "Mehmet Akif Ersoy Caddesi Nato Kavşağı Sonrası": "Buca",
  "Mehmet Akif Ersoy Caddesi Nato Kavşağı Öncesi": "Buca",
  "Mehmet Akif Ersoy Caddesi Yeşildere Katılımı Öncesi": "Buca",
  "Mehmet Akif Ersoy Caddesi İzban Önü": "Buca",
  "Mehmet Akif Ersoy Caddesi Şirinyer İzban Hizası": "Buca",
  "Mehmetçik Bulvarı Fahrettin Altay Meydanı": "Karabağlar",
  "Meles Yol Ayrımı DGM Köprü Girişi": "Konak",
  "Meles Yol Ayrımı DGM Köprü Öncesi": "Konak",
  "Meles Yol Ayrımı Dgm Köprü İnişi": "Konak",
  "Meles Yol Ayrımı Egemak Köprü Öncesi": "Konak",
  "Meles Yol Ayrımı Karşıyaka-Bornova Yol Ayrımı": "Konak",
  "Meles Yol Ayrımı Liman Ayrımı": "Konak",
  "Meles Yol Ayrımı Liman Ayrımı Öncesi": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Köprü Sonrası": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Sonrası": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Öncesi": "Konak",
  "Menderes Caddesi Belediye Kavşağı": "Buca",
  "Menderes Caddesi Heykel Kavşağı": "Buca",
  "Menderes Caddesi Heykel Meydanı": "Buca",
  "Menderes Caddesi Şirinyer Kavşağı": "Buca",
  "Milli Kütüphane Caddesi Varyant İnişi": "Konak",
  "Mimar Sinan Caddesi No:28 Önü": "Konak",
  "Mithatpaşa Caddesi 98 Sokak Girişi": "Konak",
  "Mithatpaşa Caddesi Eşref Bitlik Alt Geçit İçi": "Balçova",
  "Mithatpaşa Caddesi Eşref Bitlisi Alt Geçit İçi": "Balçova",
  "Mithatpaşa Caddesi Fahrettin Altay Kavşak İçi": "Karabağlar",
  "Mithatpaşa Caddesi Fahrettin Altay Merdan": "Karabağlar",
  "Mithatpaşa Caddesi Fahrettin Altay Meydan": "Karabağlar",
  "Mithatpaşa Caddesi Fahrettin Altay Meydanı": "Karabağlar",
  "Mithatpaşa Caddesi Göztepe Durağı Önü": "Konak",
  "Mithatpaşa Caddesi Hilal Köprü Üstü": "Konak",
  "Mithatpaşa Caddesi Karataş Kavşağı Öncesi": "Konak",
  "Mithatpaşa Caddesi Maltepe Kavşak İçi": "Narlıdere",
  "Mithatpaşa Caddesi Mehmetçik Bulvarı Girişi": "Karabağlar",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçir Çıkışı": "Balçova",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçit Çıkışı": "Balçova",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçit İçi": "Balçova",
  "Mithatpaşa Caddesi Nizamiye Önü": "Karabağlar",
  "Mithatpaşa Caddesi Otoban Kavşağı": "Balçova",
  "Mithatpaşa Caddesi Çağdaş Caddesi Kavşağı": "Balçova",
  "Mustafa Kemal Atatürk Bulvarı Marina Kavşak İçi": "Konak",
  "Mustafa Kemal Caddesi Stadyum Kavşağı": "Bornova",
  "Mustafa Kemal Caddesi Özkanlar Kavşak İçi": "Bornova",
  "Mustafa Kemal Caddesi Özkanlar Kavşağı": "Bornova",
  "Mustafa Kemal Caddesi Özkanlar Migros Kavşağı": "Bornova",
  "Mustafa Kemal Sahil Bulvarı Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Asma Köprü Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Depo Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Depo Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Feribot İskelesi Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Fevzipaşa Bulvarı Bağlantısı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Altı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Köprü Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Köprü Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Üst Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzalyalı Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı KAvşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı İç Kesime Dönüş": "Konak",
  "Mustafa Kemal Sahil Bulvarı Hava Eğitim Komutanlığı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Hava Eğitim Komutanlığı Önü": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Girişi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Çıkış": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Çıkışı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit İSonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Sor": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Çıkışı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Pier Üst Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Tünelinden Bağlantı Noktası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Viyadük Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Viyadük Üstü": "Konak",
  "Mustafa Kemal Sahil Bulvarı Köprü Durağı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Köprü Durağı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Tramvay Durağı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşak Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Sonraso": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Otoban Çıkışı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Sabancı Kavşak İçi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Sabancı Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Çeşme Otoban Bağlantısı": "Konak",
  "Mürselpaşa Bulvarı Bozkurt Kavşak İçi": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü Altı": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü Giri": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü Girişi": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü girişi": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü Çıkışı": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı DGM Köprü üstü": "Konak",
  "Mürselpaşa Bulvarı Dgm Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı Halkapınar Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı Hilal Köprü Girişi": "Konak",
  "Mürselpaşa Bulvarı Hilal Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı Hilal Köprü üstü": "Konak",
  "Mürselpaşa Bulvarı Liman Yol Ayrımı": "Konak",
  "Mürselpaşa Bulvarı Marina Kavşak İçi": "Konak",
  "Mürselpaşa Bulvarı Meles Deltası": "Konak",
  "Mürselpaşa Bulvarı Meles Fabrikası Hizası": "Konak",
  "Mürselpaşa Bulvarı Tepecik Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı Yeşildere Ayrımı Sonrası": "Konak",
  "Naldöken Köprü": "Karşıyaka",
  "Naldöken Köprü Naldöken Köprü": "Karşıyaka",
  "Nursultan Nazarbayev Caddesi Adnan Kahveci Köprü Altı": "Bayraklı",
  "Nursultan Nazarbayev Caddesi Smyrna Meydanı": "Bayraklı",
  "Onat Caddesi Homeros Kavşak İçi": "Buca",
  "Ordu Bulvarı Demir Köprü Kavşağı": "Karşıyaka",
  "Ozan Abay Caddesi Smyrna Kavşağı Öncesi": "Konak",
  "Sakarya Caddesi 252 sk. Kesişimi": "Bayraklı",
  "Sakarya Caddesi Pehlivanoğlu Kavşak İçi": "Bayraklı",
  "Sarnıç Atatürk Caddesi Sarnıç Kavşağı Öncesi": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Altı": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Çıkışı": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Öncesi": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Üstü": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü İnişi": "Gaziemir",
  "Sarnıç Atatürk Caddesi Yeşil Benzinlik Öncesi": "Gaziemir",
  "Talatpaşa Bulvarı Kıbrıs Şehitleri Caddesi Kavşağı": "Konak",
  "Uçanyol Kavşağı Homeros Bulvarı": "Konak",
  "Uçanyol Kavşağı Konak Tüneli Girişi": "Konak",
  "Uçanyol Kavşağı Merkez Ayrımı": "Konak",
  "Uğur Mumcu Caddesi Menderes Caddesi kesişimi": "Buca",
  "Yalı Bulvarı Alaybey Kavşağı": "Karşıyaka",
  "Yalı Bulvarı Alaybey Tersane Kavşak İçi": "Karşıyaka",
  "Yalı Bulvarı Muammer Aksoy Parkı Hizası": "Karşıyaka",
  "Yalı Bulvarı Muhammer Aksoy Parkı Hizası": "Karşıyaka",
  "Yalı Bulvarı Tersane Kavşağı": "Karşıyaka",
  "Yeni Girne Bulvarı Girne Pazaryeri Öncesi": "Bayraklı",
  "Yeni Girne Bulvarı Yeni Girne Kavşağı": "Bayraklı",
  "Yeni Girne Bulvarı Yeni Girne Pazaryeri Hizası": "Bayraklı",
  "Yeni Girne Bulvarı Yeni Girne Pazaryeri Sonrası": "Bayraklı",
  "Yeni Girne Bulvarı Yeni Girne Viyadük Öncesi": "Bayraklı",
  "Yeşildere Caddesi Ballıkuyu Üst Geçit Altı": "Konak",
  "Yeşildere Caddesi Besaş Üst Geçit Altı": "Konak",
  "Yeşildere Caddesi Besaş Üstgeçit Altı": "Konak",
  "Yeşildere Caddesi Buca Köprü Altı": "Konak",
  "Yeşildere Caddesi Buca Köprü Üstü": "Konak",
  "Yeşildere Caddesi DGM Köprü Girişi": "Konak",
  "Yeşildere Caddesi DGM Köprü Üstü": "Konak",
  "Yeşildere Caddesi Gaziler Caddesi Bağlantısı": "Konak",
  "Yeşildere Caddesi Halide Edip Adıvar Caddesi": "Konak",
  "Yeşildere Caddesi Halide Edip Adıvar Caddesi Katılımı": "Konak",
  "Yeşildere Caddesi Hilal Köprü Çıkışı": "Konak",
  "Yeşildere Caddesi Hilal Köprü Üstü": "Konak",
  "Yeşildere Caddesi Hilal Köprü Üstü Mürsel Paşa Katılımı": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Atatürk Köprülü Kavşak İçi": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Atatürk Köprülü Kavşağı": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Atatürk Köprülü Kavşağı Öncesi": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Köprülü Kavşak Altı": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Köprülü Kavşağı Altı": "Konak",
  "Yeşildere Caddesi Tepeci Köprü Girişi": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Basmane Katılımı": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Girişi": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Sonrası": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Öncesi": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Üstü": "Konak",
  "Yeşildere Caddesi Tepecik Köprü Üstü Basmane Yol Ayrımı": "Konak",
  "Yeşildere Caddesi Tepecil Köprü Girişi": "Konak",
  "Yeşildere Caddesi Uçanyol Katılımı Öncesi": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşak": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşak Altı": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşağı": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşağı Altı": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşağı Sonrası": "Konak",
  "Yeşildere Caddesi Uçanyol Kavşağı Öncesi": "Konak",
  "Yeşildere Caddesi Uçanyol Köprü Altı": "Konak",
  "Yeşillik Caddesi Aktepe Üst Geçit Altı": "Karabağlar",
  "Yeşillik Caddesi Bozyaka Ayrımı Öncesi": "Karabağlar",
  "Yeşillik Caddesi Buca Köprü Üstü": "Karabağlar",
  "Yeşillik Caddesi Hasan Hüseyinler Üst Geçit Altı": "Karabağlar",
  "Yeşillik Caddesi PTT Hizası": "Karabağlar",
  "Yeşillik Caddesi Paşa Köprü Lambalar Sonrası": "Karabağlar",
  "Yeşillik Caddesi Ptt Hizası": "Karabağlar",
  "Yeşillik Caddesi Serbest Bölge Alt Geçit Öncesi": "Karabağlar",
  "Yeşillik Caddesi Uçanyol Kavşağı": "Karabağlar",
  "Yeşillik Caddesi Üçyol Ayrımı Öncesi": "Karabağlar",
  "Yüzbaşı İbrahim Hakkı Caddesi 57. Topçu Tugayı Kavşak İçi": "Bayraklı",
  "Yüzbaşı İbrahim Hakkı Caddesi Smyrna Kavşak İçi": "Bayraklı",
  "Yüzbaşı İbrahim Hakkı Caddesi Smyrna Kavşağı İçi": "Bayraklı",
  "Çevreyolu Balçova Viyadüğü": "Balçova",
  "Çevreyolu Bornova Viyadüğü": "Bornova",
  "Çevreyolu Cengizhan Katılımı": "Bayraklı",
  "Çevreyolu OGM Orman Hizası": "Bornova",
  "Çevreyolu Otogar Ayrımı": "Bornova",
  "Çevreyolu Otogar Hizası": "Bornova",
  "Çevreyolu Yeşilova Viyadüğü": "Bornova",
  "Çevreyolu Örnekköy Ayrımı": "Karşıyaka",
  "Çevreyolu Örnekköy Kavşak İçi": "Karşıyaka",
  "Üniversite Caddesi Mahvel Kavşak İçi": "Bornova",
  "Üniversite Caddesi Üniversite Kavşak İçi": "Bornova",
  "İkiçeşmelik Caddesi Bayramyeri Üst Geçit Üstü": "Konak",
  "İnönü Caddesi Altıntaş Durağı Hizası": "Konak",
  "İnönü Caddesi Bayramyeri Alt Geçit İçi": "Konak",
  "İnönü Caddesi Bayramyeri Kavşağı": "Konak",
  "İnönü Caddesi Denizmen Kavşak İçi": "Konak",
  "İnönü Caddesi Fahrettin Altay Meydanı": "Karabağlar",
  "İnönü Caddesi Fahrettin Altay Meydanı KAvşak İçi": "Karabağlar",
  "İnönü Caddesi Göztepe Kavşağı": "Konak",
  "İnönü Caddesi Göztepe Stadı Sonrası": "Konak",
  "İnönü Caddesi Hıfzıssıha Kavşak İçi": "Konak",
  "İnönü Caddesi Nokta Kavşağı": "Konak",
  "İnönü Caddesi Poligon Kavşak İçi": "Konak",
  "İnönü Caddesi Yeşilyurt Kavşak İçi": "Karabağlar",
  "İnönü Caddesi Üçyol Kavşak İçi": "Konak",
  "İnönü Caddesi İslam Enstitü Kavşağı": "Konak",
  "İnönü Caddesi İslam Enstitüsü Kavşak İçi": "Konak",
  "İstanbul Caddesi 57. Topçu Tugayı Kavşağı": "Bornova",
  "İstanbul Caddesi Osman Kibar Kavşak Hizası": "Bornova",
  "İstanbul Caddesi Osman Kibar Kavşağı Öncesi": "Bornova",
  "İstanbul Caddesi Osman Kibar Tüneli SoÖncesi": "Bornova",
  "İstanbul Caddesi Osman Kibar Tüneli Öncesi": "Bornova",
  "İstanbul Caddesi Topçu Kaynağı Kavşağı": "Bornova",
  "İstanbul Caddesi Topçu Tugay Kavşağı": "Bornova",
  "İstanbul Caddesi Topçu Tugay Kavşağı Öncesi": "Bornova",
  "İstanbul Caddesi Topçu Tugayı Kavşak İçi": "Bornova",
  "İstanbul Caddesi Topçu Tugayı Kavşağı": "Bornova",
  "İstanbul Caddesi Topçu Tugayı Kavşağı Sonrası": "Bornova",
  "İstanbul Caddesi Topçu Tugayı Kavşağı Öncesi": "Bornova",
  "İstanbul Caddesi Toğçu Tugayı Hizası": "Bornova",
  "İzmir Çevre Otoyolu İStihkam Yokuşu": "Narlıdere",
  "Şair Eşref Bulvarı Eski İtfaiye Hizası": "Konak",
  "Şair Eşref Bulvarı Eski İtfaiye Öncesi": "Konak",
  "Şair Eşref Bulvarı Gazi Bulvarı Kesişimi": "Konak",
  "Şair Eşref Bulvarı Lozan Kavşağı Öncesi": "Konak",
  "Şehit Süleyman Ergin Caddesi Şehit Süleyman Ergin Caddesi": "Gaziemir",
  "Şehitler Caddesi 1. Sanayi Girişi": "Konak",
  "Şehitler Caddesi 1. Sanayi KAvşak İçi": "Konak",
  "Şehitler Caddesi 1. Sanayi Kavşak İçi": "Konak",
  "Şehitler Caddesi 1.Sanayi Kavşak İçi": "Konak",
  "Şehitler Caddesi Alsancak Meles Ayrımı": "Konak",
  "Şehitler Caddesi DGM Köprü Altı": "Konak",
  "Şehitler Caddesi DGM Köprü Girişi": "Konak",
  "Şehitler Caddesi DGM Köprü Üstü": "Konak",
  "Şehitler Caddesi Egemak Köprü Üstü": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Kavşak İçi": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Köprü Girişi": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Köprü Üstü": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Sonrası": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Öncesi": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Önü": "Konak",
  "Şehitler Caddesi Halkapınar Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Halkapınar Köprü Altı Çınarlı Dönüşü": "Konak",
  "Şehitler Caddesi Halkapınar Köprü Girişi": "Konak",
  "Şehitler Caddesi Halkapınar Köprü Üstü": "Konak",
  "Şehitler Caddesi Halkpınar Kavşak İçi": "Konak",
  "Şehitler Caddesi Halkıpınar Köprü Üstü": "Konak",
  "Şehitler Caddesi Kemer Alt Geçit İçi": "Konak",
  "Şehitler Caddesi Meles Altınyol Katılımı Öncesi": "Konak",
  "Şehitler Caddesi Meles Katılımı Öncesi": "Konak",
  "Şehitler Caddesi Mürselpaşa Caddesi Bağlantısı": "Konak",
  "Şehitler Caddesi Paralı Köprü Üstü": "Konak",
  "Şehitler Caddesi Sanayi Kavşak İçi": "Konak",
  "Şehitler Caddesi Tepecik Köprü Altı": "Konak",
  "Şehitler Caddesi Vakıflar KAvşağı": "Konak",
  "Şehitler Caddesi Vakıflar Kavşak İçi": "Konak",
  "Şehitler Caddesi Vakıflar Kavşağı": "Konak",
  "Şehitler Caddesi Şehitler Benzinlil Öncesi": "Konak",
    "Altınyol Caddesi DGM Köprü Girişi": "Konak",
    "Ankara Caddesi Zafer Payzın Köprü Üstü":"Konak",
    "Yeşillik Caddesi Konak Belediyesi Öncesi":"Konak",
    "Yeşildere Caddesi Buca Katılımı Sonrası": "Konak",
    "Girne Bulvarı Girne Pazaryeri Öncesi":"Karşıyaka",
    "Gaziler Caddesi Tepecik Köprü Öncesi":"Konak",

}
# 1) Figure out which rows are still not valid after the first pass
#    (we'll reuse ilces_set, norm, TARGET_COL, QUERY_COL already defined above)
still_not_valid_mask_initial = ~df[TARGET_COL].astype(str).map(norm).isin(ilces_set)

# 2) Apply exact mapping by y_query for the remaining rows
mapped_series = df.loc[still_not_valid_mask_initial, QUERY_COL].map(add_ilce)

# rows that will be updated by this mapping
to_fill_mask = still_not_valid_mask_initial & mapped_series.notna()

# keep a copy for logging
before_yquery_map = df[TARGET_COL].astype(str).copy()
df.loc[to_fill_mask, TARGET_COL] = mapped_series.loc[to_fill_mask]

# 3) Logging / small report
num_filled_by_yquery = int((before_yquery_map != df[TARGET_COL]).sum())

print("\n────────── y_query MAPPING ──────────")
print(f"🧭 Filled by y_query→ilçe mapping: {num_filled_by_yquery}")

# 4) Save helper CSVs
# Rows filled via y_query dict
yquery_mapped_rows = df.loc[to_fill_mask].copy()
yquery_mapped_rows.to_csv("yquery_mapped_rows.csv", index=False)

# Rows STILL not valid after applying the dict
still_not_valid_mask_final = ~df[TARGET_COL].astype(str).map(norm).isin(ilces_set)
df.loc[still_not_valid_mask_final].to_csv("not_found_ilce_after_yquery.csv", index=False)

print("💾 Saved helper files:")
print("   - yquery_mapped_rows.csv")
print("   - not_found_ilce_after_yquery.csv")

# === FINAL CLEANUP: drop unused columns, rename target to ILCE, and save ===
cols_to_drop = ["y_query", "y_first_name", "y_first_address", "y_status", "_CADDE_PLUS_KONUM"]
existing_cols = [c for c in cols_to_drop if c in df.columns]
if existing_cols:
    df.drop(columns=existing_cols, inplace=True)

# Rename the target column to ILCE (handle both possible source names)
if "ILCE" not in df.columns:
    for cand in [TARGET_COL, "y_first_addr_part2", "y_first_addr_part"]:
        if cand in df.columns and cand != "ILCE":
            df.rename(columns={cand: "ILCE"}, inplace=True)
            break

# Save final cleaned file
df.to_csv(out_path, index=False)

print("\n🧹 Dropped columns:", ", ".join(existing_cols) if existing_cols else "(none present)")
print("🏷️  Renamed target column to 'ILCE'.")
print(f"✅ Final saved to: {out_path}")

