# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:13:24 2023

@author: Kirko
Lösung zu PML CNN
Theorieaufgaben
"""
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image, ImageShow

# Aufgabe 3.2: 2D-Faltung
A = np.array([ [1, 7, 0, 1],
               [6, 1, 8, 2],
               [9, 2, 4, 1],
               [1, 0, 2, 3]])

kernel = np.array([ [1, 0, 2],
                    [0, 3, 0],
                    [2, 0, 1]])#/9

A_test = np.array([ [0,   0,   0],
                    [0, 255,   0],
                    [0,   0, 255]])

kernel_test = np.array([[0, 1, 1],
                        [1, 1, 0],
                        [0, 1, 1]])#/6


B = convolve2d(A, kernel, mode="same")
print("Matrix A: \n", A)
print("Kernel: \n", kernel)
print("Matrix B=A*kernel: \n", B)


###############################################
# Aufgabe 3.3: Filterkerne
# C = Image.open("4.png")
C = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Mittelwertfilter, Weichzeichner
kernel_1 = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])/9

# kernel_1 = np.array([ [1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1]])/25


# vertikaler Tiefpass
kernel_2 = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]])/3

# kernel_2 = np.array([ [0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0]])/5

# horizontaler Tiefpass
kernel_3 = np.array([ [0, 0, 0],
                      [1, 1, 1],
                      [0, 0, 0]])/3

# kernel_3 = np.array([ [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0],
#                       [1, 1, 1, 1, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])/5

# Hochpass, Kantendetektor
kernel_4 = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]])

# kernel_4 = np.array([ [0,  0,  -1,  0,  0],
#                       [0, -1,  -2, -1,  0],
#                       [-1, -2, 16, -2, -1],
#                       [0, -1,  -2, -1,  0],
#                       [0,  0,  -1,  0,  0]])

D = convolve2d(C, kernel_1, mode="same")
E = convolve2d(C, kernel_2, mode="same")
F = convolve2d(C, kernel_3, mode="same")
G = convolve2d(C, kernel_4, mode="same")

fig = plt.figure(figsize=(15,10))
fig.suptitle("verschiedene Filter auf Ziffer 4 angewandt",fontsize="xx-large")

plt.subplot(2,4,1)
plt.axis("off")
plt.imshow(C, cmap="gray")
plt.title("Ziffer 4 org.")

plt.subplot(2,4,5)
plt.imshow(D, cmap="gray")
plt.axis("off")
plt.title("2D-Tiefpass")

plt.subplot(2,4,2)
plt.axis("off")
plt.imshow(C, cmap="gray")
plt.title("Ziffer 4 org.")

plt.subplot(2,4,6)
plt.imshow(E, cmap="gray")
plt.axis("off")
plt.title("vertikaler Tiefpass")

plt.subplot(2,4,3)
plt.axis("off")
plt.imshow(C, cmap="gray")
plt.title("Ziffer 4 org.")

plt.subplot(2,4,7)
plt.imshow(F, cmap="gray")
plt.axis("off")
plt.title("horizontaler Tiefpass")

plt.subplot(2,4,4)
plt.axis("off")
plt.imshow(C, cmap="gray")
plt.title("Ziffer 4 org.")

plt.subplot(2,4,8)
plt.imshow(G, cmap="gray")
plt.axis("off")
plt.title("Hochpass")












