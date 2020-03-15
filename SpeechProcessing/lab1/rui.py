# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:46:45 2020

@author: hackermoon
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

F1 = [208, 374, 480, 335, 549, 805, 412, 425, 567]
F2 = [2289, 1759, 1912, 1953, 1614, 1206, 932, 1014, 1041]

label = ["i", "@", "E", "e", "6", "a", "u", "o", "O"]

import matplotlib.pyplot as plt

for i,type in enumerate(label):
    x = F1[i]
    y = F2[i]
    plt.scatter(x, y, marker='x', color='red')
    plt.text(x+10, y+10, type, fontsize=9)
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Vowel triangle")
plt.show()
plt.savefig('./other.png')
plt.close()
