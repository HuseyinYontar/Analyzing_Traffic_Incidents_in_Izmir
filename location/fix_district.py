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
  "Anadolu Caddesi Alaybey Ayrımı": "Konak",
  "Anadolu Caddesi Alaybey Çıkışı": "Gaziemir",
  "Anadolu Caddesi Ayyıldız Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Ayyıldız Kavşağı": "Bayraklı",
  "Anadolu Caddesi Ayyıldız Kavşağı Öncesi": "Bayraklı",
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
  "Anadolu Caddesi Cumhuriyet Çıkışı": "Gaziemir",
  "Anadolu Caddesi DGM": "Bayraklı",
  "Anadolu Caddesi DGM Köprü Girişi": "Bayraklı",
  "Anadolu Caddesi DGM Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi DGM Köprü İnişi": "Bayraklı",
  "Anadolu Caddesi Dedebaşı Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Dedebaşı Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Ege Deniz Bölge Komutanlığı Önü": "Gaziemir",
  "Anadolu Caddesi Ege Üniversitesi Karşıyaka Yerleşkesi": "Bornova",
  "Anadolu Caddesi Ege Üniversitesi Karşıyaka Yerleşkesi Hizası": "Bornova",
  "Anadolu Caddesi Evka 5 Kavşağı Sonrası": "Bayraklı",
  "Anadolu Caddesi Evka 5 Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Evka 5 Köprülü Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Evka 5 Öncesi": "Konak",
  "Anadolu Caddesi Ferdi Varol Yaya Üst Geçit Altı": "Bayraklı",
  "Anadolu Caddesi Gaziemir Alt Geçit İçi": "Gaziemir",
  "Anadolu Caddesi Girne Bulvarı": "Bayraklı",
  "Anadolu Caddesi Girne Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Girne Kavşağı Viyadükaltı": "Bayraklı",
  "Anadolu Caddesi Girne Köprü Katılımı Sonrası": "Bayraklı",
  "Anadolu Caddesi Girne Köprü Sonrası": "Bayraklı",
  "Anadolu Caddesi Girne Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi Girne Köprüsü": "Bayraklı",
  "Anadolu Caddesi Girne Köprüsü Sonrası": "Bayraklı",
  "Anadolu Caddesi Girne Köprüsü Öncesi": "Bayraklı",
  "Anadolu Caddesi Harmandalı Alt Geçit Öncesi": "Bayraklı",
  "Anadolu Caddesi Harmandalı Alt Geçit İçi": "Bayraklı",
  "Anadolu Caddesi Harmandalı Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Karya Evleri Hizası": "Bayraklı",
  "Anadolu Caddesi Katlı Pazaryeri": "Konak",
  "Anadolu Caddesi Katlı Pazaryeri Hizası": "Bayraklı",
  "Anadolu Caddesi Katlı Pazaryeri Karşısı": "Konak",
  "Anadolu Caddesi Katlı Pazaryeri Öncesi": "Konak",
  "Anadolu Caddesi Maltep Kavşağı": "Bayraklı",
  "Anadolu Caddesi Maltepe Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Maltepe Kavşağı": "Bayraklı",
  "Anadolu Caddesi Maltepe Kavşağı Sonrası": "Bayraklı",
  "Anadolu Caddesi Maltepe Kavşağı Öncesi": "Konak",
  "Anadolu Caddesi Meles Deltası": "Çiğli",
  "Anadolu Caddesi Naldöken Köprsü Sonrası": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü Hizası": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü Sonrası": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü SÖncesi": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü sonrası": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü Üncesi": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprü Üzeri": "Bayraklı",
  "Anadolu Caddesi Naldöken Köprüsü Öncesi": "Bayraklı",
  "Anadolu Caddesi Naldöken Yaya Geçidi Sonrası": "Bayraklı",
  "Anadolu Caddesi Naldöken köprü Sonrası": "Bayraklı",
  "Anadolu Caddesi Sarnıç Alt Geçit İçi": "Konak",
  "Anadolu Caddesi Serinkuyu Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Shell Benzinlik Hizası": "Çiğli",
  "Anadolu Caddesi Smyrna Yol Ayrımı": "Konak",
  "Anadolu Caddesi Soğukkuyu Benzinlik Önü": "Çiğli",
  "Anadolu Caddesi Soğukkuyu Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Soğukkuyu Kavşağı Hizası": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı Sonrası": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı Öncesi": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Kavşağı öncesi": "Bayraklı",
  "Anadolu Caddesi Soğukkuyu Üst Geçit Altı": "Bayraklı",
  "Anadolu Caddesi Tepecik Hastanesi Karşıyaka Semt Pol. Hizası": "Bayraklı",
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
  "Anadolu Caddesi Yeni Girne Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Yeni Girne Köprü Sonrası": "Bayraklı",
  "Anadolu Caddesi Zafer Payzın Kavşak İçi": "Konak",
  "Anadolu Caddesi Zafer Payzın Köprü Öncesi": "Konak",
  "Anadolu Caddesi Ziya Gökalp Kültür Merkezi Öncesi": "Konak",
  "Anadolu Caddesi Çevreyolu Girne Bulvarı": "Gaziemir",
  "Anadolu Caddesi Çiğli Alt Geçit": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit Hizası": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli Alt Geçit İçi": "Bayraklı",
  "Anadolu Caddesi Çiğli Kavşağı Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli Maltepe Işıkları": "Çiğli",
  "Anadolu Caddesi Çiğli Vergi Dairesi Hizası": "Çiğli",
  "Anadolu Caddesi Çiğli Vergi Dairesi Öncesi": "Çiğli",
  "Anadolu Caddesi Çiğli İtfaiye Sonrası": "Çiğli",
  "Anadolu Caddesi Çiğli İtfaiye Önü": "Çiğli",
  "Anadolu Caddesi İlave Şerit": "Konak",
  "Anadolu Caddesi İlave Şerit Çıkışı": "Konak",
  "Anadolu Caddesi Şemikler Kavşak İçi": "Gaziemir",
  "Anadolu Caddesi Şemikler Kavşağı": "Bayraklı",
  "Anadolu Caddesi Şemikler Kavşağı Sonrası": "Bayraklı",
  "Anadolu Caddesi Şemikler Kavşağı Öncesi": "Konak",
  "Anadolu Caddesi Şemikler Köprü Öncesi": "Bayraklı",
  "Anadolu Caddesi Şoğukkuyu Kavşak İçi": "Gaziemir",
  "Ankara Caddesi DGM Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Egemak Köorü Üstü": "Bayraklı",
  "Ankara Caddesi Egemak Köprü Altı": "Konak",
  "Ankara Caddesi Egemak Köprü Girişi": "Konak",
  "Ankara Caddesi Egemak Köprü Çıkışı": "Gaziemir",
  "Ankara Caddesi Egemak Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Fatih Caddesine Dönüş": "Konak",
  "Ankara Caddesi Hilal Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Mahvel Kavşağı Hizası": "Gaziemir",
  "Ankara Caddesi Mahvel Kavşağı Sonrası": "Konak",
  "Ankara Caddesi Mahvel Kavşağı Öncesi": "Konak",
  "Ankara Caddesi Manisa Kavşak İçi": "Gaziemir",
  "Ankara Caddesi Meles Katılımı Hizası": "Konak",
  "Ankara Caddesi Naldöken Kavşağı Viyadük Üstü": "Bornova",
  "Ankara Caddesi Nilüfer Kavşağı Viyadük Üstü": "Bornova",
  "Ankara Caddesi Nilüfer Kavşağı Öncesi": "Konak",
  "Ankara Caddesi Osman Kibar Alt Geçit Altı": "Konak",
  "Ankara Caddesi Osman Kibar Kavşağı Sonrası": "Konak",
  "Ankara Caddesi Osman Kibar Kavşağı Öncesi": "Konak",
  "Ankara Caddesi Osman Kibar Tünel Çıkışı": "Bornova",
  "Ankara Caddesi Osman Kibar Tüneli Çıkışı": "Bornova",
  "Ankara Caddesi Semt Garajı Hizası": "Gaziemir",
  "Ankara Caddesi Turan Köprü Hizası": "Bayraklı",
  "Ankara Caddesi Turan Köprü Üstü": "Bayraklı",
  "Ankara Caddesi Yağ Fabrikası Hizası": "Çiğli",
  "Ankara Caddesi Yağ Fabrikası Öncesi": "Konak",
  "Ankara Caddesi Zafer Payzın Köprü Girişi": "Konak",
  "Ankara Caddesi Zafer Payzın Köptü Üstü": "Konak",
  "Ankara Caddesi Zafer Payzın Meles Katılımı Öncesi": "Konak",
  "Ankara Caddesi Çevreyeyolu Ayrımı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Ayrımı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Bağlantı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Girişi": "Konak",
  "Ankara Caddesi Çevreyolu Katılı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Katılımı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Köprü Altı": "Gaziemir",
  "Ankara Caddesi Çevreyolu Köprüsü Sonrası": "Gaziemir",
  "Ankara Caddesi Çevreyolu Çıkışı": "Gaziemir",
  "Ankara Caddesi Çevreyolu İnişi": "Konak",
  "Ankara Caddesi Özkanlar Köprü Üstü": "Bayraklı",
  "Atatürk Caddesi Cennet Vadisi Hizası": "Gaziemir",
  "Atatürk Caddesi Gar Kavşağı": "Konak",
  "Atatürk Caddesi Hal Kavşak İçi": "Gaziemir",
  "Atatürk Caddesi Sebze Hali Kavşak İçi": "Gaziemir",
  "Aydın Hatboyu Caddesi Namık Kemal Caddesi Kesişimi": "Bayraklı",
  "Aydınlar Caddesi Işıkkent Çevreyolu Viyadük Altı": "Bornova",
  "Basmane Meydan Meydan": "Çiğli",
  "Caher Dudayev Bulvarı AVM Önü": "Karşıyaka",
  "Caher Dudayev Bulvarı Otoyol Gelişi": "Karşıyaka",
  "Caher Dudayev Bulvarı Otoyol Girişi": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu Ayrımı": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu Bağlantı Sonrası": "Karşıyaka",
  "Caher Dudayev Bulvarı Çevreyolu İnişi": "Karşıyaka",
  "Cemal Gürsel Caddesi Alaybey Tersane Hizası": "Bayraklı",
  "Cemal Gürsel Caddesi Anıt Hizası": "Gaziemir",
  "Cemal Gürsel Caddesi Atilla İlhan Anıtı Hizası": "Konak",
  "Cemal Gürsel Caddesi Bostanlı İskele Öncesi": "Bayraklı",
  "Cemal Gürsel Caddesi Bostanlı İskelesi Öncesi": "Bayraklı",
  "Cemal Gürsel Caddesi Muammer Aksoy Parkı Hizası": "Konak",
  "Cemal Gürsel Caddesi Naldöken Köprü Üstü": "Bayraklı",
  "Cemal Gürsel Caddesi Naldöken Köprü İnişi": "Bayraklı",
  "Cemal Gürsel Caddesi Yelken Kulubü Hizası": "Gaziemir",
  "Cemal Gürsel Caddesi İskele Hizası": "Gaziemir",
  "Cengizhan Caddesi Ege Üniversitesi Konukevi Hizası": "Bornova",
  "Cumhuriyet Bulvarı Borsa Kavşak İçi": "Buca",
  "Cumhuriyet Bulvarı Borsa Kavşağı": "Konak",
  "Cumhuriyet Bulvarı Cumhuriyet Meydanı Kavşağı": "Buca",
  "Cumhuriyet Bulvarı Fevzipaşa Bulvarı Girişi": "Konak",
  "Cumhuriyet Bulvarı Konak Pier Yaya Üst Geçit Altı": "Karşıyaka",
  "Doğuş Caddesi Tınavtepe Kavşağı": "Bayraklı",
  "Doğuş Caddesi Tınaztepe Kampüs Girişi": "Bayraklı",
  "Doğuş Caddesi Tınaztepe Kavşak İçi": "Gaziemir",
  "Doğuş Caddesi Tınaztepe Kavşağı": "Bayraklı",
  "Doğuş Caddesi Özel Hastane Kavşak İçi": "Gaziemir",
  "Eski İzmir Caddesi Köstence Köprü Öncesi": "Bayraklı",
  "Eski İzmir Caddesi Köstence Köprü Üstü": "Bayraklı",
  "Eşrefpaşa Caddesi Agora Kavşağı Sonrası": "Konak",
  "Eşrefpaşa Caddesi Bayramyeri Kavşağı": "Konak",
  "Eşrefpaşa Caddesi Buca köprü Altı": "Bayraklı",
  "Eşrefpaşa Caddesi Eski İzmir Caddesi Kesişimi": "Çiğli",
  "Eşrefpaşa Caddesi Kestelli Caddesi Girişi": "Gaziemir",
  "Eşrefpaşa Caddesi Kılıcı Mescit Hizası": "Konak",
  "Fatih Caddesi 3.Sanayi Sitesi": "Konak",
  "Fatih Caddesi Egemak Kavşak İçi": "Gaziemir",
  "Fatih Caddesi Egemak Kavşağı": "Gaziemir",
  "Fatih Caddesi Egemak Köprü Altı": "Konak",
  "Fatih Caddesi Egemak Köprü altı": "Konak",
  "Fatih Caddesi Stadyum Hizası": "Gaziemir",
  "Fatih Caddesi Vakıflar Kavşak İçi": "Gaziemir",
  "Fatih Caddesi Vakıflar Kavşağı": "Konak",
  "Fevzi Çakmak Caddesi 161 Sokak Kesişimi": "Konak",
  "Fevzipaşa Bulvarı Basmana Gar Önü": "Konak",
  "Fevzipaşa Bulvarı Basmane Gar Önü": "Konak",
  "Fevzipaşa Caddesi Konak Üst Geçit Altı": "Bayraklı",
  "Gazi Atatürk Bulvarı 73 Sokak Kesişimi": "Karşıyaka",
  "Gazi Bulvarı Borsa KAvşağı": "Konak",
  "Gazi Bulvarı Borsa Kavşak İçi": "Buca",
  "Gazi Bulvarı Borsa Kavşağı": "Konak",
  "Gazi Bulvarı Eski İtfaiye Kavşak İçi": "Gaziemir",
  "Gazi Bulvarı Halit Ziya Bulvarı Kesişimi": "Karşıyaka",
  "Gazi Bulvarı İtfaiye Kavşak İçi": "Buca",
  "Gazi Bulvarı İtfaiye Kavşağı Sonrası": "Konak",
  "Gazi Bulvarı İtfaiye Kavşağı Öncesi": "Çiğli",
  "Gazi Bulvarı Şair Eşref Bulvarı Kesişimi": "Karşıyaka",
  "Gaziler Caddesi Boğaziçi Kavşak İçi": "Gaziemir",
  "Gaziler Caddesi Kemer Alt Geçit Altı": "Gaziemir",
  "Gaziler Caddesi Kemer Alt Geçit Girişi": "Gaziemir",
  "Gaziler Caddesi Kemer Alt Geçit Çıkışı": "Gaziemir",
  "Gaziler Caddesi Kemer Alt Geçit İ": "Gaziemir",
  "Gaziler Caddesi Tepecik Köprü Altı": "Konak",
  "Gaziler Caddesi Tepecik Köprü Üstü": "Konak",
  "Gaziler Caddesi Tepecik Üstgeçit Altı": "Konak",
  "Gaziler Caddesi Yeşildere Viyadük Altı": "Bornova",
  "Gaziosmanpaşa Bulvarı İtfaiye Kavşak İçi": "Buca",
  "Girne Bulvarı Girne Kavşak İçi": "Gaziemir",
  "Girne Bulvarı Girne Kavşağı Öncesi": "Gaziemir",
  "Girne Bulvarı Girne Köprü Sonrası": "Konak",
  "Girne Bulvarı Girne Köprü Üstü": "Bayraklı",
  "Girne Bulvarı Girne Köprüsü": "Konak",
  "Girne Bulvarı Girne Kültür Park": "Konak",
  "Girne Bulvarı Girne Pazar Alanı": "Konak",
  "Girne Bulvarı Girne Viyadük Üstü": "Bornova",
  "Girne Bulvarı Girne-Günsazak Bulvarı Kesişimi": "Konak",
  "Girne Bulvarı Karaksan Kavşak İçi": "Buca",
  "Girne Bulvarı Lunapark Kavşak İçi": "Buca",
  "Girne Bulvarı Ordu Bulvarı Kavşak İçi": "Buca",
  "Girne Bulvarı Soğukkuyu Kavşağı Öncesi": "Konak",
  "Girne Bulvarı Yeni Girne Kavşak İçi": "Gaziemir",
  "Girne Bulvarı Yeni Girne Kavşağı": "Konak",
  "Girne Bulvarı Yeni Girne Kavşağı Sonrası": "Konak",
  "Girne Bulvarı Yeni Girne Köprü Sonrası": "Konak",
  "Girne Bulvarı Yeni Girne Köprü Üstü": "Bayraklı",
  "Girne Bulvarı Yeni Girne Viyadük Öncesi": "Konak",
  "Girne Bulvarı Yeni Girne Viyadük Üstü": "Konak",
  "Girne Bulvarı Özel Hastane Kavşağı": "Konak",
  "Halide Edip Adıvar Caddesi Buca Katılımı Sonrası": "Gaziemir",
  "Halide Edip Adıvar Caddesi Buca Köprü": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprü Sonrası": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprü Çıkışı": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprü Öncesi": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprü Üstü": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprü İnişi": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Köprüsü Öncesi": "Bayraklı",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı": "Karşıyaka",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı Sonrası": "Gaziemir",
  "Halide Edip Adıvar Caddesi Buca Yol Ayrımı Öncesi": "Konak",
  "Halide Edip Adıvar Caddesi Buca, Karabağlar Ayrımı Öncesi": "Konak",
  "Halide Edip Adıvar Caddesi DGM Köprü Öncesi": "Bayraklı",
  "Halide Edip Adıvar Caddesi Halide Edip Adıvar Caddesi Katılımı": "Konak",
  "Halide Edip Adıvar Caddesi Köstence Kavşağı Öncesi": "Konak",
  "Halide Edip Adıvar Caddesi Köstence Köprü Sonrası": "Konak",
  "Halide Edip Adıvar Caddesi Köstence Köprü Öncesi": "Bayraklı",
  "Halide Edip Adıvar Caddesi Köstence Üst Geçit Sonrası": "Bayraklı",
  "Halide Edip Adıvar Caddesi Yeşildere Caddesi'ne Katılım": "Konak",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı": "Konak",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı Kavşak İçi": "Gaziemir",
  "Halide Edip Adıvar Caddesi Yeşillik Caddesi Katılımı Öncesi": "Konak",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı": "Karşıyaka",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı SÖncesi": "Konak",
  "Halide Edip Adıvar Caddesi Üçyol Kavşağı Öncesi": "Konak",
  "Halide Edip Adıvar Caddesi Üçyol Meydanı": "Karşıyaka",
  "Halide Edip Adıvar Caddesi Şelale Hizası": "Çiğli",
  "Halide Edip Adıvar Caddesi Şoförler ve Otomobilciler Esnaf Odası Hizası": "Konak",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durakları Hizası": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durakları Önü": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Bahri Baba Otobüs Durağı Hizası": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durakları Hizası": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durakları Öncesi": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Bahribaba Otobüs Durağı Hizası": "Karşıyaka",
  "Halil Rıfat Paşa Caddesi Oyuncak Müzesi Öncesi": "Karşıyaka",
  "Halit Ziya Bulvarı Gazi Bulvarı Kesişimi": "Karşıyaka",
  "Hasan Ali Yücel Bulvarı Bostanlı Pazaryeri Öncesi": "Konak",
  "Hasan Ali Yücel Bulvarı Mavişehir Girişi": "Bornova",
  "Haydar Aliyev Bulvarı Marina Kavşak Öncesi": "Karşıyaka",
  "Haydar Aliyev Bulvarı Marina Kavşak İçi": "Gaziemir",
  "Haydar Aliyev Bulvarı Marina Kavşağı": "Karşıyaka",
  "Haydar Aliyev Bulvarı Marina Kavşağı Öncesi": "Karşıyaka",
  "Haydar Aliyev Bulvarı Marina Kavşağı İçinde": "Karşıyaka",
  "Haydar Aliyev Caddesi Folkart Kavşak İçi": "Gaziemir",
  "Homeros Bulvarı Konak Tünel Girişi": "Bornova",
  "Homeros Bulvarı Uçanyol Kavşağı": "Karşıyaka",
  "Homeros Bulvarı Uçanyol Kavşağı Sonrası": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı Yeşildere Katılımı": "Konak",
  "Homeros Bulvarı Uçanyol Kavşağı Öncesi": "Gaziemir",
  "Kamil Tunca Caddesi 1. Sanayi Kavşak İçi": "Gaziemir",
  "Kamil Tunca Caddesi Abdi İpekçi Caddesi Kesişimi": "Çiğli",
  "Kamil Tunca Caddesi Vakıflar Kavşak İçi": "Gaziemir",
  "Kaynak Caddesi Aydın Hatboyu Köprü Altı": "Bayraklı",
  "Konak Tüneli Konak Tüneli Girişi": "Bornova",
  "Konak Tüneli T2": "Bornova",
  "Konak Tüneli Tünel Girişi": "Bornova",
  "Liman Caddesi Liman Köprü Üstü": "Bayraklı",
  "Liman Caddesi Liman Viyadük Üstü": "Bornova",
  "Manas Bulvarı Adliye Kavşak İçi": "Gaziemir",
  "Mehmet Akif Ersoy Caddesi Beyazıt Aykut Parkı": "Karşıyaka",
  "Mehmet Akif Ersoy Caddesi Buca Köprü Altı": "Konak",
  "Mehmet Akif Ersoy Caddesi Buca köprü Altı": "Konak",
  "Mehmet Akif Ersoy Caddesi Kızılçullu Su Kemerleri Hizası": "Gaziemir",
  "Mehmet Akif Ersoy Caddesi Nato Kavşak İçi": "Gaziemir",
  "Mehmet Akif Ersoy Caddesi Nato Kavşağı Sonrası": "Konak",
  "Mehmet Akif Ersoy Caddesi Nato Kavşağı Öncesi": "Konak",
  "Mehmet Akif Ersoy Caddesi Yeşildere Katılımı Öncesi": "Konak",
  "Mehmet Akif Ersoy Caddesi İzban Önü": "Konak",
  "Mehmet Akif Ersoy Caddesi Şirinyer İzban Hizası": "Çiğli",
  "Mehmetçik Bulvarı Fahrettin Altay Meydanı": "Çiğli",
  "Meles Yol Ayrımı DGM Köprü Girişi": "Bayraklı",
  "Meles Yol Ayrımı DGM Köprü Öncesi": "Bayraklı",
  "Meles Yol Ayrımı Dgm Köprü İnişi": "Bayraklı",
  "Meles Yol Ayrımı Egemak Köprü Öncesi": "Bayraklı",
  "Meles Yol Ayrımı Karşıyaka-Bornova Yol Ayrımı": "Konak",
  "Meles Yol Ayrımı Liman Ayrımı": "Gaziemir",
  "Meles Yol Ayrımı Liman Ayrımı Öncesi": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Köprü Sonrası": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Sonrası": "Konak",
  "Meles Yol Ayrımı Zafer Payzın Öncesi": "Konak",
  "Menderes Caddesi Belediye Kavşağı": "Konak",
  "Menderes Caddesi Heykel Kavşağı": "Konak",
  "Menderes Caddesi Heykel Meydanı": "Konak",
  "Menderes Caddesi Şirinyer Kavşağı": "Konak",
  "Milli Kütüphane Caddesi Varyant İnişi": "Konak",
  "Mimar Sinan Caddesi No:28 Önü": "Konak",
  "Mithatpaşa Caddesi 98 Sokak Girişi": "Konak",
  "Mithatpaşa Caddesi Eşref Bitlik Alt Geçit İçi": "Gaziemir",
  "Mithatpaşa Caddesi Eşref Bitlisi Alt Geçit İçi": "Gaziemir",
  "Mithatpaşa Caddesi Fahrettin Altay Kavşak İçi": "Gaziemir",
  "Mithatpaşa Caddesi Fahrettin Altay Merdan": "Karşıyaka",
  "Mithatpaşa Caddesi Fahrettin Altay Meydan": "Karşıyaka",
  "Mithatpaşa Caddesi Fahrettin Altay Meydanı": "Çiğli",
  "Mithatpaşa Caddesi Göztepe Durağı Önü": "Konak",
  "Mithatpaşa Caddesi Hilal Köprü Üstü": "Bayraklı",
  "Mithatpaşa Caddesi Karataş Kavşağı Öncesi": "Konak",
  "Mithatpaşa Caddesi Maltepe Kavşak İçi": "Gaziemir",
  "Mithatpaşa Caddesi Mehmetçik Bulvarı Girişi": "Gaziemir",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçir Çıkışı": "Gaziemir",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçit Çıkışı": "Gaziemir",
  "Mithatpaşa Caddesi Mithatpaşa Alt Geçit İçi": "Gaziemir",
  "Mithatpaşa Caddesi Nizamiye Önü": "Çiğli",
  "Mithatpaşa Caddesi Otoban Kavşağı": "Karşıyaka",
  "Mithatpaşa Caddesi Çağdaş Caddesi Kavşağı": "Karşıyaka",
  "Mustafa Kemal Atatürk Bulvarı Marina Kavşak İçi": "Buca",
  "Mustafa Kemal Caddesi Stadyum Kavşağı": "Karşıyaka",
  "Mustafa Kemal Caddesi Özkanlar Kavşak İçi": "Gaziemir",
  "Mustafa Kemal Caddesi Özkanlar Kavşağı": "Konak",
  "Mustafa Kemal Caddesi Özkanlar Migros Kavşağı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Asma Köprü Sonrası": "Bayraklı",
  "Mustafa Kemal Sahil Bulvarı Depo Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Depo Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Feribot İskelesi Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Fevzipaşa Bulvarı Bağlantısı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Altı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Asma Köprü Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Köprü Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Göztepe Köprü Öncesi": "Bayraklı",
  "Mustafa Kemal Sahil Bulvarı Göztepe Üst Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzalyalı Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı KAvşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı Öncesi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Güzelyalı Kavşağı İç Kesime Dönüş": "Konak",
  "Mustafa Kemal Sahil Bulvarı Hava Eğitim Komutanlığı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Hava Eğitim Komutanlığı Önü": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Girişi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Çıkış": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Çıkışı": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit Öncesi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit İSonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karantina Alt Geçit İçi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Sor": "Konak",
  "Mustafa Kemal Sahil Bulvarı Karataş Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Çıkışı": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit Öncesi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Konak Alt Geçit İçi": "Gaziemir",
  "Mustafa Kemal Sahil Bulvarı Konak Pier Üst Geçit Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Tünelinden Bağlantı Noktası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Viyadük Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Konak Viyadük Üstü": "Bornova",
  "Mustafa Kemal Sahil Bulvarı Köprü Durağı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşağı Sonrası": "Karşıyaka",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Kavşağı Öncesi": "Karşıyaka",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Köprü Durağı Hizası": "Karşıyaka",
  "Mustafa Kemal Sahil Bulvarı Küçükyalı Tramvay Durağı Hizası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşak Öncesi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşak İçi": "Buca",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Sonraso": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Sonrası": "Konak",
  "Mustafa Kemal Sahil Bulvarı Marina Kavşağı Öncesi": "Konak",
  "Mustafa Kemal Sahil Bulvarı Otoban Çıkışı": "Konak",
  "Mustafa Kemal Sahil Bulvarı Sabancı Kavşak İçi": "Buca",
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
  "Mürselpaşa Bulvarı Marina Kavşak İçi": "Buca",
  "Mürselpaşa Bulvarı Meles Deltası": "Konak",
  "Mürselpaşa Bulvarı Meles Fabrikası Hizası": "Konak",
  "Mürselpaşa Bulvarı Tepecik Köprü Üstü": "Konak",
  "Mürselpaşa Bulvarı Yeşildere Ayrımı Sonrası": "Konak",
  "Naldöken Köprü": "Bayraklı",
  "Naldöken Köprü Naldöken Köprü": "Bayraklı",
  "Nursultan Nazarbayev Caddesi Adnan Kahveci Köprü Altı": "Bayraklı",
  "Nursultan Nazarbayev Caddesi Smyrna Meydanı": "Bayraklı",
  "Onat Caddesi Homeros Kavşak İçi": "Gaziemir",
  "Ordu Bulvarı Demir Köprü Kavşağı": "Bayraklı",
  "Ozan Abay Caddesi Smyrna Kavşağı Öncesi": "Bayraklı",
  "Sakarya Caddesi 252 sk. Kesişimi": "Konak",
  "Sakarya Caddesi Pehlivanoğlu Kavşak İçi": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Kavşağı Öncesi": "Bayraklı",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Altı": "Bayraklı",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Çıkışı": "Gaziemir",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Öncesi": "Bayraklı",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü Üstü": "Bayraklı",
  "Sarnıç Atatürk Caddesi Sarnıç Köprü İnişi": "Bayraklı",
  "Sarnıç Atatürk Caddesi Yeşil Benzinlik Öncesi": "Gaziemir",
  "Talatpaşa Bulvarı Kıbrıs Şehitleri Caddesi Kavşağı": "Konak",
  "Uçanyol Kavşağı Homeros Bulvarı": "Konak",
  "Uçanyol Kavşağı Konak Tüneli Girişi": "Bornova",
  "Uçanyol Kavşağı Merkez Ayrımı": "Konak",
  "Uğur Mumcu Caddesi Menderes Caddesi kesişimi": "Gaziemir",
  "Yalı Bulvarı Alaybey Kavşağı": "Konak",
  "Yalı Bulvarı Alaybey Tersane Kavşak İçi": "Buca",
  "Yalı Bulvarı Muammer Aksoy Parkı Hizası": "Karşıyaka",
  "Yalı Bulvarı Muhammer Aksoy Parkı Hizası": "Karşıyaka",
  "Yalı Bulvarı Tersane Kavşağı": "Konak",
  "Yeni Girne Bulvarı Girne Pazaryeri Öncesi": "Gaziemir",
  "Yeni Girne Bulvarı Yeni Girne Kavşağı": "Konak",
  "Yeni Girne Bulvarı Yeni Girne Pazaryeri Hizası": "Karşıyaka",
  "Yeni Girne Bulvarı Yeni Girne Pazaryeri Sonrası": "Gaziemir",
  "Yeni Girne Bulvarı Yeni Girne Viyadük Öncesi": "Bayraklı",
  "Yeşildere Caddesi Ballıkuyu Üst Geçit Altı": "Bayraklı",
  "Yeşildere Caddesi Besaş Üst Geçit Altı": "Bayraklı",
  "Yeşildere Caddesi Besaş Üstgeçit Altı": "Bayraklı",
  "Yeşildere Caddesi Buca Köprü Altı": "Konak",
  "Yeşildere Caddesi Buca Köprü Üstü": "Bayraklı",
  "Yeşildere Caddesi DGM Köprü Girişi": "Konak",
  "Yeşildere Caddesi DGM Köprü Üstü": "Bayraklı",
  "Yeşildere Caddesi Gaziler Caddesi Bağlantısı": "Konak",
  "Yeşildere Caddesi Halide Edip Adıvar Caddesi": "Konak",
  "Yeşildere Caddesi Halide Edip Adıvar Caddesi Katılımı": "Konak",
  "Yeşildere Caddesi Hilal Köprü Çıkışı": "Konak",
  "Yeşildere Caddesi Hilal Köprü Üstü": "Bayraklı",
  "Yeşildere Caddesi Hilal Köprü Üstü Mürsel Paşa Katılımı": "Konak",
  "Yeşildere Caddesi Mustafa Kemal Atatürk Köprülü Kavşak İçi": "Gaziemir",
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
  "Yeşildere Caddesi Uçanyol Köprü Altı": "Bayraklı",
  "Yeşillik Caddesi Aktepe Üst Geçit Altı": "Gaziemir",
  "Yeşillik Caddesi Bozyaka Ayrımı Öncesi": "Konak",
  "Yeşillik Caddesi Buca Köprü Üstü": "Bayraklı",
  "Yeşillik Caddesi Hasan Hüseyinler Üst Geçit Altı": "Gaziemir",
  "Yeşillik Caddesi PTT Hizası": "Gaziemir",
  "Yeşillik Caddesi Paşa Köprü Lambalar Sonrası": "Bayraklı",
  "Yeşillik Caddesi Ptt Hizası": "Gaziemir",
  "Yeşillik Caddesi Serbest Bölge Alt Geçit Öncesi": "Gaziemir",
  "Yeşillik Caddesi Uçanyol Kavşağı": "Gaziemir",
  "Yeşillik Caddesi Üçyol Ayrımı Öncesi": "Konak",
  "Yüzbaşı İbrahim Hakkı Caddesi 57. Topçu Tugayı Kavşak İçi": "Gaziemir",
  "Yüzbaşı İbrahim Hakkı Caddesi Smyrna Kavşak İçi": "Gaziemir",
  "Yüzbaşı İbrahim Hakkı Caddesi Smyrna Kavşağı İçi": "Gaziemir",
  "Çevreyolu Balçova Viyadüğü": "Gaziemir",
  "Çevreyolu Bornova Viyadüğü": "Buca",
  "Çevreyolu Cengizhan Katılımı": "Konak",
  "Çevreyolu OGM Orman Hizası": "Konak",
  "Çevreyolu Otogar Ayrımı": "Gaziemir",
  "Çevreyolu Otogar Hizası": "Gaziemir",
  "Çevreyolu Yeşilova Viyadüğü": "Gaziemir",
  "Çevreyolu Örnekköy Ayrımı": "Gaziemir",
  "Çevreyolu Örnekköy Kavşak İçi": "Buca",
  "Üniversite Caddesi Mahvel Kavşak İçi": "Gaziemir",
  "Üniversite Caddesi Üniversite Kavşak İçi": "Gaziemir",
  "İkiçeşmelik Caddesi Bayramyeri Üst Geçit Üstü": "Bayraklı",
  "İnönü Caddesi Altıntaş Durağı Hizası": "Bayraklı",
  "İnönü Caddesi Bayramyeri Alt Geçit İçi": "Gaziemir",
  "İnönü Caddesi Bayramyeri Kavşağı": "Konak",
  "İnönü Caddesi Denizmen Kavşak İçi": "Gaziemir",
  "İnönü Caddesi Fahrettin Altay Meydanı": "Çiğli",
  "İnönü Caddesi Fahrettin Altay Meydanı KAvşak İçi": "Buca",
  "İnönü Caddesi Göztepe Kavşağı": "Konak",
  "İnönü Caddesi Göztepe Stadı Sonrası": "Konak",
  "İnönü Caddesi Hıfzıssıha Kavşak İçi": "Gaziemir",
  "İnönü Caddesi Nokta Kavşağı": "Konak",
  "İnönü Caddesi Poligon Kavşak İçi": "Gaziemir",
  "İnönü Caddesi Yeşilyurt Kavşak İçi": "Gaziemir",
  "İnönü Caddesi Üçyol Kavşak İçi": "Gaziemir",
  "İnönü Caddesi İslam Enstitü Kavşağı": "Bayraklı",
  "İnönü Caddesi İslam Enstitüsü Kavşak İçi": "Gaziemir",
  "İstanbul Caddesi 57. Topçu Tugayı Kavşağı": "Konak",
  "İstanbul Caddesi Osman Kibar Kavşak Hizası": "Gaziemir",
  "İstanbul Caddesi Osman Kibar Kavşağı Öncesi": "Bayraklı",
  "İstanbul Caddesi Osman Kibar Tüneli SoÖncesi": "Bornova",
  "İstanbul Caddesi Osman Kibar Tüneli Öncesi": "Bornova",
  "İstanbul Caddesi Topçu Kaynağı Kavşağı": "Gaziemir",
  "İstanbul Caddesi Topçu Tugay Kavşağı": "Bayraklı",
  "İstanbul Caddesi Topçu Tugay Kavşağı Öncesi": "Bayraklı",
  "İstanbul Caddesi Topçu Tugayı Kavşak İçi": "Gaziemir",
  "İstanbul Caddesi Topçu Tugayı Kavşağı": "Konak",
  "İstanbul Caddesi Topçu Tugayı Kavşağı Sonrası": "Bayraklı",
  "İstanbul Caddesi Topçu Tugayı Kavşağı Öncesi": "Bayraklı",
  "İstanbul Caddesi Toğçu Tugayı Hizası": "Gaziemir",
  "İzmir Çevre Otoyolu İStihkam Yokuşu": "Gaziemir",
  "Şair Eşref Bulvarı Eski İtfaiye Hizası": "Konak",
  "Şair Eşref Bulvarı Eski İtfaiye Öncesi": "Çiğli",
  "Şair Eşref Bulvarı Gazi Bulvarı Kesişimi": "Karşıyaka",
  "Şair Eşref Bulvarı Lozan Kavşağı Öncesi": "Konak",
  "Şehit Süleyman Ergin Caddesi Şehit Süleyman Ergin Caddesi": "Konak",
  "Şehitler Caddesi 1. Sanayi Girişi": "Konak",
  "Şehitler Caddesi 1. Sanayi KAvşak İçi": "Gaziemir",
  "Şehitler Caddesi 1. Sanayi Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi 1.Sanayi Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Alsancak Meles Ayrımı": "Konak",
  "Şehitler Caddesi DGM Köprü Altı": "Konak",
  "Şehitler Caddesi DGM Köprü Girişi": "Konak",
  "Şehitler Caddesi DGM Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Egemak Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Halkapınar Aktarma Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Halkapınar Aktarma Köprü Girişi": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Halkapınar Aktarma Sonrası": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Öncesi": "Konak",
  "Şehitler Caddesi Halkapınar Aktarma Önü": "Konak",
  "Şehitler Caddesi Halkapınar Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Halkapınar Köprü Altı Çınarlı Dönüşü": "Bayraklı",
  "Şehitler Caddesi Halkapınar Köprü Girişi": "Konak",
  "Şehitler Caddesi Halkapınar Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Halkpınar Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Halkıpınar Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Kemer Alt Geçit İçi": "Gaziemir",
  "Şehitler Caddesi Meles Altınyol Katılımı Öncesi": "Konak",
  "Şehitler Caddesi Meles Katılımı Öncesi": "Konak",
  "Şehitler Caddesi Mürselpaşa Caddesi Bağlantısı": "Konak",
  "Şehitler Caddesi Paralı Köprü Üstü": "Bayraklı",
  "Şehitler Caddesi Sanayi Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Tepecik Köprü Altı": "Konak",
  "Şehitler Caddesi Vakıflar KAvşağı": "Konak",
  "Şehitler Caddesi Vakıflar Kavşak İçi": "Gaziemir",
  "Şehitler Caddesi Vakıflar Kavşağı": "Konak",
  "Şehitler Caddesi Şehitler Benzinlil Öncesi": "Konak",
    "Altınyol Caddesi DGM Köprü Girişi": "Konak",
    "Ankara Caddesi Zafer Payzın Köprü Üstü":"Konak",
    "Yeşillik Caddesi Konak Belediyesi Öncesi":"Konak",
    "Yeşildere Caddesi Buca Katılımı Sonrası": "Buca",
    "Girne Bulvarı Girne Pazaryeri Öncesi":"Bayraklı",
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

