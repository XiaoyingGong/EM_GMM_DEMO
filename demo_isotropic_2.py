# author: 龚潇颖(Xiaoying Gong)
# date： 2020/7/12 10:04  
# IDE：PyCharm 
# des: 各向同性的协方差,mu固定住
# input(s)：
# output(s)：
import numpy as np
import time
from scipy import stats
import matplotlib.pyplot as plt
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
    return generated_data.T, Z

# 各向同性的高斯
# data为n*dim的
# mu为1*dim的
# sigma为scalar
def gaussian_isotropic(data, mu, sigma):
  dim = data.shape[1]
  A = 1 / (np.sqrt(2 * np.pi) * sigma)**dim
  B = np.exp((-1/2) * (np.linalg.norm(data - mu, axis=1)/sigma) ** 2)
  return A * B

# 高斯混合模型
def gaussian_mixture_model(data, mus, sigmas, alphas):
    k = len(alphas) #高斯的个数
    pd = 0
    for i in range(k):
        pd += alphas[i] * gaussian_isotropic(data, mus[i], sigmas[i])
    return pd

# 期望 计算响应度
def Expectation(data, mus, sigmas, alphas, N, K):
    response = np.zeros([N, K])
    for k in range(K):
        response[:, k] = gaussian_isotropic(data, mus[k], sigmas[k]) * alphas[k]
    response_sum = np.sum(response, axis=1).reshape(-1, 1)
    response = response / response_sum
    return response

# 最大化 对各个值求偏导
def Maximization(data, mus, response, N, K):
    dim = data.shape[1]
    # 初始化
    # mus = np.zeros([K, dim])
    sigmas = np.zeros(K)
    # 更新alpha
    alphas = (1/N) * np.sum(response, axis=0)
    # 更新mu
    for k in range(K):
        # mus[k, :] = np.sum(data * response[:, k].reshape([-1, 1]), axis=0) / np.sum(response[:, k])
        zero_mean_dist = np.linalg.norm(data - mus[k, :], axis=1) ** 2
        sigmas[k] = np.sum(response[:, k] * zero_mean_dist) / np.sum(response[:, k])
        sigmas[k] = np.sqrt((1/dim) * sigmas[k])
        if sigmas[k] <= 0:#防止sigma为奇异值（cov是奇异矩阵即np.linalg.det(cov) == 0，这儿就是sigma=0）
            sigmas[k] = 0.00001
    return sigmas, alphas

# EM过程
# initial_mu:k*dim, initial_sigma:scalar, initial_alpha:(k,~)
def EM_algorithm(data, initial_mu, initial_sigma, initial_alpha, max_iterative, is_draw=True):
    dim = initial_mu.shape[1]
    N = len(data)
    K = len(initial_alpha)

    mus = initial_mu
    sigmas = initial_sigma
    alphas = initial_alpha
    if is_draw:
        draw_em(data, mus, sigmas, alphas, -1)
    for i in range(max_iterative):
        # E-step
        response = Expectation(data, mus, sigmas, alphas, N, K)
        # M-step
        sigmas, alphas = Maximization(data,mus, response, N, K)
        if is_draw:
            draw_em(data, mus, sigmas, alphas, i)
    return mus, sigmas, alphas

def draw_em(data, mus, sigmas, alphas, i):
    if i == -1:
        print("初始状态:", "\nmus:\n", mus, "\nsigmas:\n", sigmas, "\nalphas:\n", alphas)
        figurename = "初始状态"
    else:
        print("当前第",i+1,"次迭代:", "\nmus:\n", mus, "\nsigmas:\n", sigmas, "\nalphas:\n", alphas)
        figurename = "当前第" + str(i+1) + "次迭代"
    # 绘制
    x_show, y_show = np.mgrid[-10:10:.1, -10:10:.1]
    x_show_flatten = x_show.flatten().reshape(-1, 1)
    y_show_flatten = y_show.flatten().reshape(-1, 1)
    gmm_pd = gaussian_mixture_model(np.hstack((x_show_flatten, y_show_flatten)), mus, sigmas, alphas)
    gmm_pd_show = gmm_pd.reshape(200, 200)
    plt.figure(figurename)
    plt.axis("equal")
    plt.contour(x_show, y_show, gmm_pd_show)
    plt.scatter(mus[:, 0], mus[:, 1], s=10, c='red')
    plt.scatter(data[:, 0], data[:, 1], s=1, c='blue')
    plt.show()

def process_1():
    # 产生数据
    mus = np.array([[-3, 4], [0, 0], [3, 4]])
    sigmas = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    num = 1000
    component_rate = np.array([0.3, 0.3, 0.4])
    data, Z = generate_gmm_data(mus, sigmas, num, component_rate)
    # 计算与更新
    initial_mus = np.array([[-2, 3], [1, 1], [2, 5]], dtype=np.float)
    initial_sigmas = np.array([0.5, 0.5, 0.5], dtype=np.float)
    initial_alpha = np.ones(len(initial_mus)) / len(initial_mus)
    start_time = time.time()
    mus_results, sigmas_results, weights_results =\
        EM_algorithm(data, initial_mus, initial_sigmas, initial_alpha, 5)
    end_time = time.time()
    print(end_time - start_time)
    return mus_results, sigmas_results, weights_results

if __name__ == '__main__':
    process_1()