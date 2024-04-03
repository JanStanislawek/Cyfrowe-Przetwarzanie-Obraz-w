# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:03:05 2021

@author: Admin
"""


# %%
# Wczytywanaie bibliotek
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Wczytywanie obrazu
image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Wczytywanie obrazu w skali szarości
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()

# %%
# Sprawdzenie typu 
type(image)

# %%
# Wyświetlenie danych obrazu
image

# %%
# Wyświetlenie rozdzielczości obrazu
image.shape

# %%
# Wyświetlenie wartości piksela – dostęp do tablicy
image[0,0]

# %%
# Wczytywanie obrazu kolorowego
image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)

# %%
type(image_bgr)

# %%
image_bgr.shape

# %%
# Konwersja na przestrzeń barw RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Wyświetlenie obrazu kolorowego
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# %%
# Zapisywanie obrazu
cv2.imwrite("images/plane_gray.jpg", image)

# %%
# Wczytanie obrazu w skali szarości
im_256x256 = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# %%
# Zmiana wielkości obrazu na 50x50 pikseli
im_50x50 = cv2.resize(image, (50, 50))

# %%
# Wyświetlenie obrazu
plt.imshow(im_50x50, cmap="gray"), plt.axis("off")

# %%
# Wczytanie obrazu z pliku
image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# %%
# Pobranie połowy obrazu – połowa kolumn i wszystkie wiersze
im_256x256_cropped = image[:,:128]

# %%
# Wyświetlenie nowego obrazu
plt.imshow(im_256x256_cropped, cmap="gray"), plt.axis("off")
