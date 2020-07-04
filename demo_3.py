# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/3 16:39  
# IDE：PyCharm 
# des:在demo_2的基础上，将协方差变成各向同性的
# input(s)：
# output(s)：
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mpl_toolkits.mplot3d

mus_for_draw = np.array([[-3, 4], [0, 0], [3, 4]])
add_4_valid = 0.01
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
    dim = mu.shape[0]
    sigma = sigma * np.identity(dim, dtype = np.float)
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
    dim = mu.shape[0]
    sigma = sigma * np.identity(dim, dtype=np.float)
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
def EM_algorithm(data, initial_mus, initial_sigmas, initial_weights, MaxIter = 20, is_draw = True):
    prev_mus = initial_mus
    prev_sigmas = initial_sigmas
    prev_weights = initial_weights

    next_mus = np.zeros(initial_mus.shape, dtype=np.float)
    next_sigmas = np.zeros(len(initial_sigmas), dtype=np.float)
    next_weights = np.zeros(initial_weights.shape, dtype=np.float)

    N = len(data[0])
    dim = len(data)
    k = len(prev_weights)
    response = np.zeros([N, k], dtype=np.float)

    for t in range(MaxIter):
        print("第", t, "次迭代")
        if t > 0:
            # 计算责任矩阵 compute the responsibilities
            for l in range(k):
                response[:, l] = gaussian(prev_mus[l], prev_sigmas[l], data)

            response_sum = np.sum(response, axis=1).reshape(-1, 1)
            response = response / response_sum

            # 更新权重 weight^(i+1)
            next_weights = np.sum(response, axis=0) / N
            # 更新均值 next_mus^(i+1) 现在就不更新了 均值不会变化
            # for l in range(k):
            #         next_mus[l, :] = np.sum(data.T * (response[:, l].reshape(-1, 1)), axis=0) / np.sum(response[:, l])

            # 更新sigma
            for l in range(k):
                zero_mean_data = data.T - next_mus[l, :].reshape(1, -1)
                covariances = np.zeros([dim, dim, N])
                for i in range(N):
                    covariances[:, :, i] = zero_mean_data[i,:].reshape(1, -1).T * zero_mean_data[i,:].reshape(1, -1) * response[i, l]
                next_sigmas_temp = np.sum(covariances, axis=2) / np.sum(response[:, l])
                next_sigmas[l] = np.sum(next_sigmas_temp.diagonal()) / (next_sigmas_temp.shape[0])
            prev_mus = next_mus
            prev_sigmas = next_sigmas
            prev_weights = next_weights
        else:
            next_mus = prev_mus
            next_sigmas = prev_sigmas
            next_weights = prev_weights
        if is_draw:

            # 绘制
            x_show, y_show = np.mgrid[-10:10:.1, -10:10:.1]
            x_show_flatten = x_show.flatten()
            y_show_flatten = y_show.flatten()
            gmm_pd = gaussian_mixture_model(next_mus, next_sigmas, next_weights,
                                            np.vstack((x_show_flatten, y_show_flatten)))
            gmm_pd_show = gmm_pd.reshape(200, 200)
            plt.figure("gmm_contour_results")
            plt.axis("equal")
            plt.scatter(mus_for_draw[:, 0], mus_for_draw[:, 1])
            plt.contour(x_show, y_show, gmm_pd_show)
            plt.scatter(next_mus[:, 0], next_mus[:, 1], s=10, c='red')
            plt.scatter(data[0, :], data[1, :], s=1)
            plt.show()
    return next_mus, next_sigmas, next_weights


# -------------------------------- 以下为测试 --------------------------------
# 对于 generate_gmm_data(mus, sigmas, num, component_rate)的测试
def generate_gmm_data_test():
    mus = np.array([[1, 1], [2, 5], [5, 5], [5, 3]])
    sigmas = np.array([[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.2]], [[0.3, 0], [0, 0.3]], [[0.1, 0], [0, 0.3]]])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.2, 0.2])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    colors = ['green', 'red', 'blue', 'orange']
    plt.scatter(data[0, :], data[1, :], s=1, c=[colors[int(Z[i]%len(colors))] for i in range(num)])
    plt.show()

# 每个生成的component中选一个点作为质心
def processing():
    # 产生数据
    mus = np.array([[-3, 4], [0, 0], [3, 4]])
    # sigmas = np.array([[[2, 0], [0, 1]], [[2, 0], [0, 1]], [[2, 0], [0, 1]]])
    sigmas = np.array([2, 2, 2])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.4])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    z_0 = np.argwhere(Z == 0)
    z_1 = np.argwhere(Z == 1)
    z_2 = np.argwhere(Z == 2)
    mu_1 = data[:, z_0[int(np.floor(len(z_0) / 2))]].flatten()
    mu_2 = data[:, z_1[int(np.floor(len(z_1) / 2))]].flatten()
    mu_3 = data[:, z_2[int(np.floor(len(z_2) / 2))]].flatten()
    # 计算与更新
    # 现在的mu是某三个点作为质心 即均值
    initial_mus = np.array([mus[0], mus[1], mus[2]], dtype=np.float)
    initial_sigmas = np.array([0.5, 0.5, 0.5], dtype=np.float)
    initial_weights = np.ones(len(initial_mus)) / len(initial_mus)
    start_time = time.time()
    mus_results, sigmas_results, weights_results = EM_algorithm(data, initial_mus, initial_sigmas, initial_weights, MaxIter=20)
    end_time = time.time()
    print(end_time - start_time)
    return mus_results, sigmas_results, weights_results


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
    # gaussian_mixture_model_test()
    mus, sigmas, weights = processing()
    print("mus:", mus, "sigmas:", sigmas, "weights:", weights)