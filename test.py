# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/3 9:10  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# 概率模型 高斯模型，n维高斯
# mu是:1行*dim维,sigma是:dim维度*dim维的数据,x是dim维*n个的数据
def gaussian(mu, sigma, data):
    dim = sigma.shape[0]
    pos = np.empty((1,) + (data.shape[1], ) + (dim,))
    for i in range(dim):
        pos[:, :, i] = data[i, :]
    # print(sigma)
    # print("行列式:", np.linalg.det(sigma))
    # if np.linalg.det(sigma) == 0.:
    #     sigma += np.array([[add_4_valid, 0.], [0., add_4_valid]])
    rv = stats.multivariate_normal(mu, sigma)
    return rv.pdf(pos)

def gaussian_test(mu, sigma, data):
    gaussian(mu, sigma, data)
    return

def my_gaussian(mu, sigma, data):

    return


def my_gaussian_test(mu, sigma, data):
    return

if __name__ == '__main__':
    x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
    print(x.flatten().reshape[1, -1])
    data = np.vstack((x.flatten().reshape[1, -1], y.flatten().reshape[1, -1]))
    mu = np.array([5, 5])
    sigma = np.array([[1, 0], [0, 1]])
    gaussian(mu, sigma, 1)