from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompajliranje
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
my_callbacks = [
    callbacks.TensorBoard(log_dir='logs'),
    callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
]

# Treniranje
history = model.fit(
    x_train_s, y_train_s,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks
)

# Ucitaj najbolji model
best_model = models.load_model('best_model.h5')

# Predikcije
y_train_pred = np.argmax(best_model.predict(x_train_s), axis=1)
y_test_pred = np.argmax(best_model.predict(x_test_s), axis=1)

y_train_true = np.argmax(y_train_s, axis=1)
y_test_true = np.argmax(y_test_s, axis=1)

# Tocnost
train_acc = accuracy_score(y_train_true, y_train_pred)
test_acc = accuracy_score(y_test_true, y_test_pred)

print("Tocnost na skupu za ucenje:", train_acc)
print("Tocnost na testnom skupu:", test_acc)

# Matrica zabune
cm = confusion_matrix(y_test_true, y_test_pred)
print("Matrica zabune (test):")
print(cm)