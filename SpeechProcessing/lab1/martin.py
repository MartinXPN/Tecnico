# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:46:45 2020

@author: hackermoon
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

F1 = [311.0234061556743, 399.8390666654584, 464.8703116281331, 494.0032955796277, 475.5939826554058, 586.8925320965703, 334.12355656377355, 480.11211482937676, 585.8954456146215]
F2 = [2636.1949101135237, 2107.955597012013, 1920.1862517962861, 1279.7396877251947, 1399.9912086696588, 971.899437958144, 824.6657903268647, 850.0039663526943, 1044.9281743417232]

label = ["i", "e", "E", "@", "6", "a", "u", "o", "O"]

for i, type in enumerate(label):
    x = F1[i]
    y = F2[i]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+10, y+10, type, fontsize=9)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Vowel triangle")
plt.show()
plt.savefig('./vtri_94771.png')
plt.close()
