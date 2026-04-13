import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("occupancy_processed.csv")
X = df.drop("Room_Occupancy_Count", axis=1)
y = df["Room_Occupancy_Count"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Stablo odlučivanja
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# 5) Predikcija
y_pred = tree.predict(X_test)

# 6) Evaluacija
print("\nMATRICA ZABUNE:")
print(confusion_matrix(y_test, y_pred))

print("\nTOČNOST:")
print(accuracy_score(y_test, y_pred))

print("\nIZVJEŠTAJ:")
print(classification_report(y_test, y_pred))

# 7) Vizualizacija stabla
plt.figure(figsize=(15, 8))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["Slobodna", "Zauzeta"],
    filled=True
)
plt.title("Stablo odlučivanja")
plt.show()


