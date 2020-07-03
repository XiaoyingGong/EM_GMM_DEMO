# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/2 9:34  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import mpl_toolkits.mplot3d

x, y = np.mgrid[-10:10:.1, -10:10:.1]
# print(x)
pos = np.empty(x.shape + (2,))  # 从x.shape=(200,200)变为(200,200,2)

pos[:, :, 0] = x
pos[:, :, 1] = y
# mean=[0.5, -0.2],cov=[[2.0, 0.3], [0.3, 0.5]]，声明一个带着指定mean和cov的rv对象
rv = multivariate_normal([0.5, -0.2], [[2.0, 0], [0, 2]])

# 将f(X,Y)=rv.pdf(pos)的值对应到color map的暖色组中寻找(X,Y)对应的点对应的颜色
the_pd = rv.pdf(pos)
plt.axis("equal")

plt.contourf(x, y, rv.pdf(pos))
plt.figure()
print(x.shape)
print(y.shape)
print(rv.pdf(pos).shape)
plt.contour(x, y, rv.pdf(pos))
# ax = plt.subplot(111, projection='3d')
# ax.set_title('www.linuxidc.com - matplotlib Code Demo');
# ax.plot_surface(x, y, the_pd, rstride=2, cstride=1, cmap=plt.cm.Spectral)
plt.show()
