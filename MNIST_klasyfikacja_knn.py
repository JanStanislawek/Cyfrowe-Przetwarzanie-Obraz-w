# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:49:39 2023

@author: Admin
"""



# %%
# 1
# Wczytanie zależności



import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split



# %%
# 2
# Ładowanie danych wariant 1
# Load data from https://www.openml.org/d/554

from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

images = X.to_numpy().reshape(70000, 28, 28)
target = y.to_numpy()

fig, axes = plt.subplots(nrows=1, ncols=4)

for ax, image, label in zip(axes, images, target):
    #ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Trening: {label}")

# Podział danych uczących na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, shuffle=True)

# %%
# 3
# Ładowanie danych wariont 2
# Load data from keras.datasets

from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()



# %%
# 4
# Standaryzacja danych

X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_st = scaler.transform(X_train)
# X_test_st = scaler.transform(X_test)

# %%
# 5.1
# Tworzenie klasyfikatorów: klasyfikator kNN

from sklearn.neighbors import KNeighborsClassifier
k=3
clf_knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', n_jobs=-1)


# %%
# 5.2
# Trenowanie klasyfikatorów

clf_knn.fit(X_train, y_train)

clf_knn.score(X_test, y_test)
# %%
# 5.3
# Predykcja klas na zbiorze testowym
predicted = clf_knn.predict(X_test)

# Wyświetla podgląd 4 pierwszych przykładów testowych
fig, axes = plt.subplots(nrows=1, ncols=4)
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(f"Predykcja: {prediction}")
    
# Wyświetla raport klasyfikacji
print(f"Raport klasyfikacji  {clf_knn}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")

# Wyświetla macierz pomyłek w oknie graficznym
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Macierz pomyłek")

# WYświetla macierz pomyłek w konsoli
print(f"Macierz pomyłek:\n{disp.confusion_matrix}")



# %%
# 6.1
# Wykonać tylko jeśli wystarczy czasu
# Tworzenie klasyfikatorów: klasyfikator Regresja logistyczna

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(C=10.0, random_state=1, solver='lbfgs', max_iter=10000)


# %%
# 6.2
# Trenowanie klasyfikatorów

clf_lr.fit(X_train, y_train)


# %%
# 6.3
# Predykcja klas na zbiorze testowym
predicted = clf_lr.predict(X_test)


fig, axes = plt.subplots(nrows=1, ncols=4)
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(f"Predykcja: {prediction}")
    
    
print(
    f"Raport klasyfikacji  {clf_lr}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Macierz pomyłek")
print(f"Macierz pomyłek:\n{disp.confusion_matrix}")







