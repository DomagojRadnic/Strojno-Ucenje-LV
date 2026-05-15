import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

filename = 'test.png'

# Ucitaj sliku
img_original = mpimg.imread('zad.png') 
img = color.rgb2gray(img_original[:, :, :3])
img = resize(img, (28, 28))

# Prikazi sliku
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')  
plt.show()

# Pripremi sliku - ulaz u mrezu
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# Ucitaj model
model = models.load_model('best_model.h5')

# Predikcija
prediction = model.predict(img)

# Rezultat
predicted_class = np.argmax(prediction)

print("Predvidena znamenka:", predicted_class)

