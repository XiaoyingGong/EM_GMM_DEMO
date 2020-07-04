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
    rv = stats.multivariate_normal(mu, sigma)
    return rv.pdf(pos)

# 现在sigma就是个scalar
def my_gaussian(mu, sigma, data):
    dim = mu.shape[0]
    pos = np.empty((1,) + (data.shape[1], ) + (dim,))
    for i in range(dim):
        pos[:, :, i] = data[i, :]
    cov_sigma = np.identity(dim, dtype=np.float)
    print(cov_sigma)
    cov_sigma = cov_sigma * sigma
    mu_re = mu.reshape([1, 1, dim])
    print(pos.shape)
    print(mu_re.shape)
    print(cov_sigma.shape)
    A = 1 / (2 * np.pi) ** (dim / 2) * (1 / np.linalg.det(cov_sigma) ** (1 / 2))
    B = np.exp((-1 / 2) * np.dot(np.dot((pos - mu_re).T, np.linalg.inv(cov_sigma)), (pos - mu_re)))
    return A * B

def gaussian_test(mu, sigma, data):

    return gaussian(mu, sigma, data)

def my_gaussian_test(mu, sigma, data):
    return my_gaussian(mu, sigma, data)



if __name__ == '__main__':
  a = np.array([[2], [2], [2]])
  b = np.array([[2, 2, 2]])
  print(np.dot(b, a))