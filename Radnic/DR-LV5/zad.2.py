import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


df = pd.read_csv("occupancy_processed.csv")
X = df.drop("Room_Occupancy_Count", axis=1)
y = df["Room_Occupancy_Count"]

# 3) Podjela (80% train, 20% test, stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Skaliranje
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 6) Predikcija
y_pred = knn.predict(X_test_scaled)

# 7) Evaluacija

print("\nMATRICA ZABUNE:")
print(confusion_matrix(y_test, y_pred))

print("\nTOČNOST:")
print(accuracy_score(y_test, y_pred))

print("\nPRECIZNOST I ODZIV:")
print(classification_report(y_test, y_pred))