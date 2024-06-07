# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:58:55 2024

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications
import matplotlib.pyplot as plt
import numpy as np

# Ładowanie danych CIFAR-100
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalizacja wartości pikseli do zakresu 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Nazwy klas CIFAR-100
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 
    'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 
    'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 
    'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 
    'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 
    'willow_tree', 'wolf', 'woman', 'worm'
]

# Tworzenie modelu sieci neuronowej ResNet50
base_model = applications.ResNet50(
    weights=None,      # Nie korzystamy z pretrenowanych wag
    input_shape=(32, 32, 3),
    classes=100
)

base_model.summary()

# %%
# Kompilacja modelu
base_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Funkcja do wyświetlania obrazów ze zbioru danych
def display_images(images, labels, predictions=None, class_names=None):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        idx = np.random.randint(0, len(images))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[idx])
        if predictions is None:
            label = labels[idx][0]
            label_name = class_names[label] if class_names else str(label)
            plt.xlabel(f"Label: {label_name}")
        else:
            label = labels[idx][0]
            prediction = np.argmax(predictions[idx])
            label_name = class_names[label] if class_names else str(label)
            prediction_name = class_names[prediction] if class_names else str(prediction)
            plt.xlabel(f"Label: {label_name}\nPred: {prediction_name}")
    plt.show()

# Wyświetlanie kilku obrazów przed trenowaniem
print("Wyświetlanie kilku obrazów ze zbioru treningowego:")
display_images(train_images, train_labels, class_names=class_names)

# Trenowanie modelu z monitorowaniem funkcji straty
history = base_model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))
# %%

# Ocena modelu
test_loss, test_acc = base_model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc}')

# Wyświetlanie funkcji straty w trakcie trenowania
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Funkcja Straty w Trakcie Trenowania')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()

# Wyświetlanie kilku obrazów z przewidywaniami modelu
print("Wyświetlanie kilku obrazów ze zbioru testowego z przewidywaniami modelu:")
predictions = base_model.predict(test_images)
display_images(test_images, test_labels, predictions, class_names=class_names)
