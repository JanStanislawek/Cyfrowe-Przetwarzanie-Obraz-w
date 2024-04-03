# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:52:02 2024

@author: Admin
"""

# import zależności
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/wiki.jpg', 0)

plt.subplot(421)
plt.imshow(img, cmap='gray')

plt.subplot(422)
plt.hist(img.ravel(), bins=256, range=[0, 256])

plt.subplot(423)
img_equ = cv2.equalizeHist(img)
plt.imshow(img_equ, cmap='gray')

plt.subplot(424)
plt.hist(img_equ.ravel(), bins=256, range=[0, 256])

plt.subplot(425)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
img_equ_clahe = clahe.apply(img)
plt.imshow(img_equ_clahe, cmap='gray')

plt.subplot(426)
plt.hist(img_equ_clahe.ravel(), bins=256, range=[0, 256])

plt.subplot(427)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
img_equ_clahe2 = clahe.apply(img_equ_clahe)
plt.imshow(img_equ_clahe2, cmap='gray')

plt.subplot(428)
plt.hist(img_equ_clahe2.ravel(), bins=256, range=[0, 256])

plt.show
