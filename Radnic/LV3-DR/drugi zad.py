import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('mtcars.csv')

plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 2, 1)
cyl_groups = df.groupby('cyl')['mpg'].mean()
bars = ax1.bar(['4 cilindra', '6 cilindara', '8 cilindara'], 
               cyl_groups.values, 
               color=['#2ecc71', '#3498db', '#e74c3c'],
               alpha=0.8,
               edgecolor='black',
               linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Prosječna potrošnja (MPG)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Broj cilindara', fontsize=11, fontweight='bold')
ax1.set_title('1. Prosječna potrošnja po broju cilindara', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = plt.subplot(2, 2, 2)
data_4cyl = df[df['cyl'] == 4]['wt'] * 1000 * 0.453592  # Pretvorim u kg
data_6cyl = df[df['cyl'] == 6]['wt'] * 1000 * 0.453592
data_8cyl = df[df['cyl'] == 8]['wt'] * 1000 * 0.453592

bp = ax2.boxplot([data_4cyl, data_6cyl, data_8cyl],
                   labels=['4 cilindra', '6 cilindara', '8 cilindara'],
                   patch_artist=True,
                   showmeans=True,
                   meanprops=dict(marker='D', markerfacecolor='red', markersize=7))

colors = ['#2ecc71', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Masa (kg)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Broj cilindara', fontsize=11, fontweight='bold')
ax2.set_title('2. Distribucija težine automobila po broju cilindara', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

ax3 = plt.subplot(2, 2, 3)

manual_mpg = df[df['am'] == 1]['mpg']
auto_mpg = df[df['am'] == 0]['mpg']

parts = ax3.violinplot([auto_mpg, manual_mpg], 
                        positions=[1, 2],
                        showmeans=True,
                        showmedians=True)

ax3.boxplot([auto_mpg, manual_mpg],
            positions=[1, 2],
            widths=0.1,
            patch_artist=True,
            boxprops=dict(facecolor='yellow', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))

ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Automatski\nmjenjač', 'Ručni\nmjenjač'])
ax3.set_ylabel('Potrošnja (MPG)', fontsize=11, fontweight='bold')
ax3.set_title('3. Potrošnja po vrsti mjenjača', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

auto_mean = auto_mpg.mean()
manual_mean = manual_mpg.mean()
ax3.text(1, auto_mean, f'{auto_mean:.2f}', ha='center', va='bottom', fontweight='bold')
ax3.text(2, manual_mean, f'{manual_mean:.2f}', ha='center', va='bottom', fontweight='bold')

# Ispis podataka
print(f"Prosječna potrošnja - Automatski mjenjač: {auto_mean:.2f} MPG")
print(f"Prosječna potrošnja - Ručni mjenjač: {manual_mean:.2f} MPG")
print(f"Razlika: {manual_mean - auto_mean:.2f} MPG\n")

ax4 = plt.subplot(2, 2, 4)

auto_data = df[df['am'] == 0]
ax4.scatter(auto_data['hp'], auto_data['qsec'], 
           s=150, alpha=0.6, c='#e74c3c', label='Automatski mjenjač',
           edgecolor='black', linewidth=1)

manual_data = df[df['am'] == 1]
ax4.scatter(manual_data['hp'], manual_data['qsec'],
           s=150, alpha=0.6, c='#3498db', label='Ručni mjenjač',
           edgecolor='black', linewidth=1, marker='^')

ax4.set_xlabel('Snaga (KS)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Ubrzanje - vrijeme 0-60 mph (sec)', fontsize=11, fontweight='bold')
ax4.set_title('4. Odnos snage i ubrzanja po vrsti mjenjača', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax4.grid(True, alpha=0.3)
ax4.invert_yaxis()  

plt.suptitle('ANALIZA MTCARS SKUPA PODATAKA - GRAFIČKI PRIKAZ', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig('mtcars_analiza_grafovi.png', dpi=300, bbox_inches='tight')
print("Slika je spremljena kao 'mtcars_analiza_grafovi.png'")

plt.show()

print("\n" + "="*80)
print("ZBIRNA ANALIZA")
print("="*80)

print("\n1. PROSJEČNA POTROŠNJA PO BROJU CILINDARA:")
for cyl in [4, 6, 8]:
    avg = df[df['cyl'] == cyl]['mpg'].mean()
    count = len(df[df['cyl'] == cyl])
    print(f"   {cyl} cilindara: {avg:.2f} MPG ({count} automobila)")

print("\n2. TEŽINA PO BROJU CILINDARA (u kg):")
for cyl in [4, 6, 8]:
    weights = df[df['cyl'] == cyl]['wt'] * 1000 * 0.453592
    print(f"   {cyl} cilindara - Min: {weights.min():.2f} kg, Max: {weights.max():.2f} kg, Sredina: {weights.median():.2f} kg")

print("\n3. POTROŠNJA PO VRSTI MJENJAČA:")
print(f"   Automatski mjenjač: {auto_mpg.mean():.2f} MPG (n={len(auto_mpg)})")
print(f"   Ručni mjenjač: {manual_mpg.mean():.2f} MPG (n={len(manual_mpg)})")
print(f"   >>> Ručni mjenjač ima bolju potrošnju za {manual_mpg.mean() - auto_mpg.mean():.2f} MPG!")

print("\n4. SNAGA I UBRZANJE PO VRSTI MJENJAČA:")
print(f"   Automatski mjenjač - Prosječna snaga: {auto_data['hp'].mean():.2f} KS, Prosječno ubrzanje: {auto_data['qsec'].mean():.2f} sec")
print(f"   Ručni mjenjač - Prosječna snaga: {manual_data['hp'].mean():.2f} KS, Prosječno ubrzanje: {manual_data['qsec'].mean():.2f} sec")

print("\n" + "="*80)
