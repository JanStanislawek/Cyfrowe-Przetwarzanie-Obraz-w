# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:14:39 2024

@author: Admin
"""

# import zależności
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread('images/fruits.jpg', )

img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
img_lab[..., 0] = clahe.apply(img_lab[..., 0])
img_equ = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

plt.rcParams['figure.figsize'] = [8, 6]

plt.subplot(221)
plt.imshow(img_bgr[...,::-1])
plt.title("BGR")

plt.subplot(222)
plt.hist(img_bgr[..., 0].ravel(), bins=256, range=[0, 256], color='b')
plt.hist(img_bgr[..., 1].ravel(), bins=256, range=[0, 256], color='g')
plt.hist(img_bgr[..., 2].ravel(), bins=256, range=[0, 256], color='r')

plt.subplot(223)
plt.imshow(img_equ[...,::-1])
plt.title("LAB")

plt.subplot(224)
plt.hist(img_equ[..., 0].ravel(), bins=256, range=[0, 256], color='b')
plt.hist(img_equ[..., 1].ravel(), bins=256, range=[0, 256], color='g')
plt.hist(img_equ[..., 2].ravel(), bins=256, range=[0, 256], color='r')

img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
img_lab[..., 0] = clahe.apply(img_lab[..., 0])
img_equ = cv2.cvtColor(img_lab, cv2.COLOR_HSV2BGR)

plt.subplot(223)
plt.imshow(img_equ[...,::-1])
plt.title("HSV")

plt.subplot(224)
plt.hist(img_equ[..., 0].ravel(), bins=256, range=[0, 256], color='b')
plt.hist(img_equ[..., 1].ravel(), bins=256, range=[0, 256], color='g')
plt.hist(img_equ[..., 2].ravel(), bins=256, range=[0, 256], color='r')

plt.show()
