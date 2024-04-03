# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:34:39 2024

@author: Admin
"""

# %%
# 1
# Wczytanie zależności i danych 

import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()



# %%
# 2 Podgląd danych

fig, axes = plt.subplots(nrows=1, ncols=4)

for ax, image, label in zip(axes, digits.images, digits.target):
    #ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(f"Trening: {label}")




# %%
# 3
# Podział danych uczących na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.5, shuffle=False)




# %%
# 4
# Standaryzacja danych

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# %%
# 5.1
# Tworzenie klasyfikatorów: klasyfikator kNN

from sklearn.neighbors import KNeighborsClassifier
k=5
clf_knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)


# %%
# 5.2
# Trenowanie klasyfikatorów

clf_knn.fit(X_train, y_train)

clf_knn.score(X_test. y_test)


# %%
# 5.3
# Predykcja klas na zbiorze testowym
predicted = clf_knn.predict(X_test)

# Wyświetla podgląd 4 pierwszych przykładów testowych
fig, axes = plt.subplots(nrows=1, ncols=4)
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(f"Predykcja: {prediction}")
    
# Wyświetla raport klasyfikacji
print(f"Raport klasyfikacji  {clf_knn}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")

# Wyświetla macierz pomyłek w oknie graficznym
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Macierz pomyłek")

# Wyświetla macierz pomyłek w konsoli
print(f"Macierz pomyłek:\n{disp.confusion_matrix}")