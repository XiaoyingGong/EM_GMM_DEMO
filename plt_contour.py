# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/2 14:36  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 3, 4, 5])

Z = np.random.random((4, 3))
X, Y = np.meshgrid(x, y)
print(X)
print(Y)
a = plt.contourf(X, Y, Z, 3, cmap=plt.cm.Spectral)
b = plt.contour(X, Y, Z, 3, colors='black', linewidths=1, linestyles='solid')
plt.show()