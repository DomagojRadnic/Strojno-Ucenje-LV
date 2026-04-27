import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Učitaj sliku
image = Image.open("example.png")
image = image.convert("RGB")
img = np.array(image)

w, h, d = img.shape

pixels = img.reshape(-1, 3)

K = 4 

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels)
centers = np.uint8(kmeans.cluster_centers_)

quantized_pixels = centers[labels]

quantized_img = quantized_pixels.reshape(w, h, 3)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika (K={K})")
plt.imshow(quantized_img)
plt.axis("off")

plt.tight_layout()
plt.show()