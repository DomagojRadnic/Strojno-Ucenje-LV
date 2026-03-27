import pandas as pd
df = pd.read_csv('mtcars.csv')
print("=" * 80)
print("ANALIZA MTCARS SKUPA PODATAKA")
print("=" * 80)

print("\n1. PET AUTOMOBILA S NAJVEĆOM POTROŠNJOM (MPG):")
print("-" * 80)
top_5_mpg = df.sort_values(by='mpg', ascending=False).head(5)
print(top_5_mpg[['car', 'mpg']].to_string(index=False))

print("\n2. TRI AUTOMOBILA S 8 CILINDARA S NAJMANJOM POTROŠNJOM:")
print("-" * 80)
eight_cyl = df[df['cyl'] == 8].sort_values(by='mpg', ascending=True).head(3)
print(eight_cyl[['car', 'mpg', 'cyl']].to_string(index=False))

print("\n3. SREDNJA POTROŠNJA AUTOMOBILA SA 6 CILINDARA:")
print("-" * 80)
avg_6cyl = df[df['cyl'] == 6]['mpg'].mean()
print(f"Srednja potrošnja: {avg_6cyl:.2f} MPG")

print("\n4. SREDNJA POTROŠNJA AUTOMOBILA S 4 CILINDRA, MASE IZMEĐU 2000-2200 LBS:")
print("-" * 80)
filtered_4cyl = df[(df['cyl'] == 4) & (df['wt'] >= 2.0) & (df['wt'] <= 2.2)]
if len(filtered_4cyl) > 0:
    avg_4cyl_filtered = filtered_4cyl['mpg'].mean()
    print(f"Automobili: {', '.join(filtered_4cyl['car'].tolist())}")
    print(f"Srednja potrošnja: {avg_4cyl_filtered:.2f} MPG")
else:
    print("Nema automobila koji zadovoljavaju kriterije.")

print("\n5. BROJ AUTOMOBILA PO VRSTI MJENJAČA:")
print("-" * 80)
manual_count = len(df[df['am'] == 1])
automatic_count = len(df[df['am'] == 0])
print(f"Ručni mjenjač (am=1): {manual_count} automobila")
print(f"Automatski mjenjač (am=0): {automatic_count} automobila")

print("\n6. AUTOMOBILI S AUTOMATSKIM MJENJAČEM I SNAGOM PREKO 100 KS:")
print("-" * 80)
auto_high_power = df[(df['am'] == 0) & (df['hp'] > 100)]
print(f"Broj automobila: {len(auto_high_power)}")
print("\nDetaljno:")
print(auto_high_power[['car', 'am', 'hp']].to_string(index=False))

print("\n7. MASA SVAKOG AUTOMOBILA U KILOGRAMIMA:")
print("-" * 80)
df_kg = df.copy()
df_kg['masa_kg'] = df_kg['wt'] * 1000 * 0.453592  
print(df_kg[['car', 'masa_kg']].to_string(index=False))
print("\nAlternativni prikaz:")
for idx, row in df_kg.iterrows():
    print(f"{row['car']:<25} {row['wt'] * 1000 * 0.453592:>8.2f} kg")
print("\n" + "=" * 80)


