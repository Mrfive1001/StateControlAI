
"""
Reward Design
Designer: Lin Cheng  2018.01.22
aa
aaaaa
aaa
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 100000)

y = np.zeros(x.shape)

b = 0.99

a = x.shape

x_penalty = 0.75
# for i in range(0, 100000):
#     if x[i] < x_penalty:
#        c
#     else:
#         xxx = (x[i] - x_penalty)/(1 - x_penalty)
#         y[i] = -np.log2(1.01 - xxx) / np.log2(b) /10


a = 1
for i in range(0, 100000):
    if x[i] < x_penalty:
        y[i] = 0
    else:
        # xxx = (x[i] - x_penalty) / (1 - x_penalty)
        y[i] = - 1000* (np.exp(0.1*(x[i]-x_penalty)) -1)








plt.figure(1)
plt.plot(x, y)
plt.grid()
plt.show()





