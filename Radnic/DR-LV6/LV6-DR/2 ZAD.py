from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            centers=4,
            cluster_std=[1.0, 2.5, 0.5, 3.0],
            random_state=random_state
        )
    elif flagc == 4:
        X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = datasets.make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

X = generate_data(500, 1)

K = range(1, 21)
J = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    J.append(kmeans.inertia_)

plt.figure()
plt.plot(K, J, marker='o')
plt.xlabel('Broj klastera (k)')
plt.ylabel('J')
plt.title('Elbow metoda')
plt.grid()
plt.show()