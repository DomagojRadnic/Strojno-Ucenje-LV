'''
Room occupancy classification 

R.Grbic, 2024.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()
print("Broj primjera:", df.shape[0])
unique, counts = np.unique(y, return_counts=True)
print("Razdioba po klasama:")
for u, c in zip(unique, counts):
    print(f"Klasa {u} ({class_names[u]}): {c}")

#1 kada je co2 veci prostorija je zauzeta
#2 Broj primjera: 10129
#3 Podaci nisu savršeno ravnomjerno raspodijeljeni po klasama



