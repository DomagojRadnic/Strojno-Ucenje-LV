import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

image = mpimg.imread('example_grayscale.png')

if len(image.shape) == 3:
    image = image[:, :, 0]

X = image.reshape((-1, 1))

k = 10
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
kmeans.fit(X)

values = kmeans.cluster_centers_.squeeze()
labels = kmeans.labels_

image_compressed = np.choose(labels, values)
image_compressed = image_compressed.reshape(image.shape)

plt.figure()
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.figure()
plt.title("Kvantizirana slika (k=10)")
plt.imshow(image_compressed, cmap='gray')
plt.axis('off')

plt.show()