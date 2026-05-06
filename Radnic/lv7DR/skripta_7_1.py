import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 1) Prikaz nekoliko slika iz train skupa
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# 2) Kreiraj mrežu pomoću keras.Sequential
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Prikaz strukture mreže
model.summary()

# 3) Definiraj karakteristike procesa učenja
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4) Provodi treniranje mreže
history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=32,
                    validation_split=0.1)

# 5) Izračunaj točnost mreže na skupu za učenje i testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

# 6) Prikaz matrice zabune
y_pred = np.argmax(model.predict(x_test_s), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrica zabune - Test skup")
plt.show()

# 7) Prikaz nekoliko pogrešno klasificiranih primjera
misclassified_idx = np.where(y_pred != y_test)[0]
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 1) Prikaz nekoliko slika iz train skupa
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# 2) Kreiraj mrežu pomoću keras.Sequential
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Prikaz strukture mreže
model.summary()

# 3) Definiraj karakteristike procesa učenja
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4) Provodi treniranje mreže
history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=32,
                    validation_split=0.1)

# 5) Izračunaj točnost mreže na skupu za učenje i testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

# 6) Prikaz matrice zabune
y_pred = np.argmax(model.predict(x_test_s), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrica zabune - Test skup")
plt.show()

# 7) Prikaz nekoliko pogrešno klasificiranih primjera
misclassified_idx = np.where(y_pred != y_test)[0]
plt.figure(figsize=(10,5))
for i, idx in enumerate(misclassified_idx[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
    plt.axis('off')
plt.show()