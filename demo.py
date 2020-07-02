# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/1 16:03  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mpl_toolkits.mplot3d

# 每个产生的高斯的个数为多少
# 产生的数据存在着一定数量的偏差，不是严格按照一定比例
# component_rate:
def create_num_of_each_gaussian(component_rate, num, is_offset = True):
    k = len(component_rate) # 计算有多少个component
    offset = np.zeros(k, dtype=int)
    if is_offset:
        offset[:k-1] = np.array([int(generate_gaussian_data(0, np.array([10]), 1).flatten()[0]) for i in range(k - 1)])
        offset[k-1] = -np.sum(offset)
    component_num = np.dot(num, component_rate) + offset
    return component_num

# 生成数据
# mu:均值, sigma:协方差 num：数据的数量 dim:数据的维度
def generate_gaussian_data(mu, sigma, num):
    dim = sigma.shape
    data = np.random.multivariate_normal(mu, sigma, num).T
    return data

# 生成gmm数据
# mus: 每个component的均值， sigmas：每个component的协方差，
# num:数据量, component_rate：每个component的比例
# 生成的数据为dim维 * num个
def generate_gmm_data(mus, sigmas, num, component_rate):
    if np.sum(component_rate) != 1:
        print("component_rate的和不为1")
        return
    dim = len(mus[0])
    k = len(component_rate)
    # 生成的数据量有一定的偏差
    each_component_num = create_num_of_each_gaussian(component_rate, num, False)
    # 生成属于每一个component的数据
    Z = np.zeros(num)
    start = 0
    # 生成类别
    for i in range(k):
        end = int(each_component_num[i])
        Z[start:start+end] = i
        start = start + end
    # 生成数据
    generated_data = np.zeros((dim, num))
    for i in range(k):
        generated_data[:, Z == i] = generate_gaussian_data(mus[i], sigmas[i], int(each_component_num[i]))
    return generated_data, Z

# 概率模型 高斯模型，n维高斯
# mu是:1行*dim维,sigma是:dim维度*dim维的数据,x是dim维*n个的数据
def gaussian(mu, sigma, data):
    dim = sigma.shape[0]
    pos = np.empty((1,) + (data.shape[1], ) + (dim,))
    for i in range(dim):
        pos[:, :, i] = data[i, :]
    rv = stats.multivariate_normal(mu, sigma)
    return rv.pdf(pos)

# 计算responsibility probability
def cal_responsibility():

    return

# 高斯混合模型
def gaussian_mixture_model(mus, sigmas, alphas, data):
    k = len(alphas) #高斯的个数
    pd = 0
    for i in range(k):
        pd += alphas[i] * gaussian(mus[i], sigmas[i], data)
    return pd

# E-M algorithm
def EM_algorithm(data, initial_mus, initial_sigmas, initial_weights):
    return

# 主函数
def processing():
    # 产生数据
    mus = np.array([[1, 1], [2, 5], [5, 5], [5, 3]])
    sigmas = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.2]], [[0.3, 0], [0, 0.3]], [[0.1, 0], [0, 0.3]]])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.2, 0.2])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    # 计算与更新
    initial_mus = np.array([[0, 1], [2, 1], [3, 1], [4, 1]])
    initial_sigmas = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    initial_weights = np.ones(len(initial_mus)) / len(initial_mus)


# -------------------------------- 以下为测试 --------------------------------
# 对于 generate_gmm_data(mus, sigmas, num, component_rate)的测试
def generate_gmm_data_test():
    mus = np.array([[1, 1], [2, 5], [5, 5], [5, 3]])
    sigmas = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.2]], [[0.3, 0], [0, 0.3]], [[0.1, 0], [0, 0.3]]])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.2, 0.2])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    plt.figure("generated_data_results")
    colors = ['green', 'red', 'blue', 'orange']
    plt.scatter(data[0, :], data[1, :], s=1, c=[colors[int(Z[i]%len(colors))] for i in range(num)])
    plt.show()

# 对于高斯混合模型的测试
def gaussian_mixture_model_test():
    mus = np.array([[1, 1], [2, 5], [5, 5], [5, 3]])
    sigmas = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.2]], [[0.3, 0], [0, 0.3]], [[0.1, 0], [0, 0.3]]])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.2, 0.2])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    print(data.shape)
    gmm_pd = gaussian_mixture_model(mus, sigmas, component_rate, data)
    # print(gmm_pd)
    plt.figure("gmm_contour_results")
    colors = ['green', 'red', 'blue', 'orange']
    print(data[0, :].shape)
    print(data[1, :].shape)
    print(gmm_pd.shape)
    plt.contour(data[0, :], data[1, :], gmm_pd)
    plt.scatter(data[0, :], data[1, :], s=1, c=[colors[int(Z[i]%len(colors))] for i in range(num)])
    # plt.scatter(data[0, :], data[], rv.pdf(pos))
    plt.show()


if __name__ == '__main__':

    # x, y = np.mgrid[-10:10:.1, -10:10:.1]
    # # print(x)
    # pos = np.empty(x.shape + (2,))  # 从x.shape=(200,200)变为(200,200,2)
    #
    # pos[:, :, 0] = x
    # pos[:, :, 1] = y
    # # mean=[0.5, -0.2],cov=[[2.0, 0.3], [0.3, 0.5]]，声明一个带着指定mean和cov的rv对象
    # rv = multivariate_normal([0.5, -0.2], [[2.0, 0], [0, 0.5]])
    # true = rv.pdf([5, 5])
    # print(true)
    # a = gaussian(np.array([0.5, -0.2]), np.array([[2.0, 0], [0, 0.5]]), np.array([[0, 0.5, 1, 2, 3, 4, 5], [0, -0.2, 1, 2, 3, 4, 5]]))
    # print(a)
    gaussian_mixture_model_test()
