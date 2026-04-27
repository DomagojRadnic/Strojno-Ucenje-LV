import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# 1. Generiranje podataka
X = generate_data(500, 1)   # mijenjaj flagc (1–5)

# -------------------------------
# 2. KMeans
k = 3   # broj klastera (po potrebi promijeni)

kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# -------------------------------
# 3. Prikaz originalnih podataka
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=20)
plt.title("Generirani podaci")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# -------------------------------
# 4. Prikaz klastera
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')

plt.title("K-means grupiranje")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

