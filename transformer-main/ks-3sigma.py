import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, nbinom
def ks():
    import matplotlib.pyplot as plt
    from scipy.stats import norm, poisson, nbinom

    # 读取数据集
    df = pd.read_csv("NYC Taxi Traffic.csv")

    # 提取 value 列作为示例数据
    data = df['value']

    # 绘制直方图
    plt.hist(data, bins=20, density=False, alpha=0.5, color='g')

    # 根据直方图观察可能的分布情况
    # 例如，可以尝试正态分布、泊松分布、负二项分布等
    # x = np.linspace(data.min(), data.max(), 1000)

    # 正态分布拟合
    # mu, std = norm.fit(data)
    # pdf_norm = norm.pdf(x, mu, std)
    # plt.plot(x, pdf_norm, 'k-', label="Normal")

    # 泊松分布拟合
    # lam = np.mean(data)
    # pdf_poisson = poisson.pmf(x, lam)
    # plt.plot(x, pdf_poisson, 'r--', label="Poisson")

    # 负二项分布拟合
    # variance = np.var(data)
    # p = 1 - (variance / lam)
    # n = lam * (1 - p) / p
    # pdf_nbinom = nbinom.pmf(x, n, p)
    # plt.plot(x, pdf_nbinom, 'b-.', label="Negative Binomial")

    plt.legend()
    plt.show()
def ksdetect():
    df = pd.read_csv("NYC Taxi Traffic.csv")

    # 提取 value 列作为示例数据
    data = df['value']

    # 计算数据的均值
    lam = data.mean()
    print('lam: ', lam)
    # 设定异常标签阈值
    threshold = lam * 1.5  # 可根据实际情况调整阈值

    # 标记异常数据
    outliers = data[data > threshold]

    # 输出异常数据
    print('outliers.size', outliers.size)
    print("异常数据：\n", outliers)

def NYC():
    from datetime import datetime
    # 读取数据集
    df = pd.read_csv("NYC Taxi Traffic.csv")
    # time_series = df['timestamp']
    # timestamps = []
    # for time_str in time_series:
    #     time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    #     timestamp = datetime.timestamp(time_obj)
    #     timestamps.append(timestamp)
    #
    # # 将时间戳添加到数据集中
    # df['timestamp'] = timestamps
    # print(df)
    # 根据条件标记异常数据
    df['label'] = 0  # 初始化标签列，所有数据先标记为0
    df.loc[(df['value'] > 30000) | (df['value'] < 1000), 'label'] = 1  # 将符合条件的数据标记为1

    # 输出包含异常标签的数据集
    print(len(df[df['label'] == 1]))
    print(df[df['label']==1])
    # print(df)

    # data = pd.DataFrame(df)
    # data.to_csv('NYC.CSV',index=False)

    # print(data)

def jihuo():
    # 定义常用激活函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    # 定义x范围
    x = np.linspace(-3.5, 3.5, 50)

    # 绘制激活函数图像
    plt.figure(figsize=(10, 6))

    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.plot(x, tanh(x), label='Tanh')
    plt.plot(x, relu(x), label='ReLU')
    plt.plot(x, elu(x), label='ELU')

    plt.title('Comparison of Activation Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.grid(True)
    plt.show()

def draw():
    # 每个数据集的具体值
    # 表4-5 gcn depths
    pems_depth_1 = [99.674, 97.556, 94.745, 98.188]
    nyc_depth_1 = [99.360, 99.648, 99.392, 99.136]

    pems_depth_2 = [99.671, 97.502, 95.720, 98.308]
    nyc_depth_2 = [99.456, 99.648, 99.360, 99.104]

    pems_depth_3 = [99.586, 97.650, 96.071, 98.550]
    nyc_depth_3 = [99.424, 99.648, 99.424, 99.008]

    # 表4-4 gnn depths
    # pems_depth_1 = [99.674, 97.556, 94.745, 98.188]
    # nyc_depth_1 = [99.360, 99.648, 99.392, 99.136]
    #
    # pems_depth_2 = [99.660,	97.449,	98.888,	98.722]
    # nyc_depth_2 = [99.456,	99.488,	98.848,	99.200]
    #
    # pems_depth_3 = [99.677,	97.470,	98.885,	98.943]
    # nyc_depth_3 = [99.360,	99.392,	99.136,	99.040]

    # K 的取值
    K_values = [2, 10, 50, 100]

    # 绘制折线图
    plt.figure(figsize=(10, 8))

    # Depth 1
    plt.plot(K_values, pems_depth_1, marker='o', linestyle='-', color='black', label='PEMS Depth 1')
    plt.plot(K_values, nyc_depth_1, marker='o', linestyle='--', color='red', label='NYC Depth 1')

    # Depth 2
    plt.plot(K_values, pems_depth_2, marker='o', linestyle='-', color='green', label='PEMS Depth 2')
    plt.plot(K_values, nyc_depth_2, marker='o', linestyle='--', color='orange', label='NYC Depth 2')

    # Depth 3
    plt.plot(K_values, pems_depth_3, marker='o', linestyle='-', color='blue', label='PEMS Depth 3')
    plt.plot(K_values, nyc_depth_3, marker='o', linestyle='--', color='yellow', label='NYC Depth 3')

    plt.xlabel('K')
    plt.ylabel('AUC')

    # 添加具体值标签
    # for i in range(len(K_values)):
    #     plt.text(K_values[i]-0.2, pems_depth_1[i]-0.2, f'{pems_depth_1[i]:.3f}', ha='center', va='bottom', fontsize=8)
    #     plt.text(K_values[i], nyc_depth_1[i] -0.1, f'{nyc_depth_1[i]:.3f}', ha='center', va='bottom', fontsize=8)
    #
    #     plt.text(K_values[i], pems_depth_2[i], f'{pems_depth_2[i]:.3f}', ha='center', va='bottom', fontsize=8)
    #     plt.text(K_values[i], nyc_depth_2[i] -0.3, f'{nyc_depth_2[i]:.3f}', ha='center', va='bottom', fontsize=8)
    #
    #     plt.text(K_values[i]+0.2, pems_depth_3[i]-0.1, f'{pems_depth_3[i]:.3f}', ha='center', va='bottom', fontsize=8)
    #     plt.text(K_values[i], nyc_depth_3[i]-0.2, f'{nyc_depth_3[i]:.3f}', ha='center', va='bottom', fontsize=8)

    plt.title('AUC scores of PEMS and NYC data sets under different gcn Depths and K values')
    plt.legend()
    plt.grid(True)
    plt.xticks(K_values)  # 设置横轴刻度
    plt.show()

# ksdetect()
# ks()
# NYC()
# jihuo()
# ziti()
# draw()

# 生成离群图
def liqun():
    # 生成类别C1的数据点
    np.random.seed(0)
    num_C1 = 400  # C1类别的数据点数量
    mean_C1 = [10, 10]  # C1类别的均值
    cov_C1 = [[3, 0], [0, 3]]  # C1类别的协方差矩阵
    C1_data = np.random.multivariate_normal(mean_C1, cov_C1, num_C1)

    # 生成类别C2的数据点
    num_C2 = 200  # C2类别的数据点数量
    mean_C2 = [-3, -3]  # C2类别的均值
    cov_C2 = [[3, 0], [0, 3]]  # C2类别的协方差矩阵
    C2_data = np.random.multivariate_normal(mean_C2, cov_C2, num_C2)

    # 生成离群点O1和O2
    O1 = [7, -2]
    O2 = [10, 0]

    # 可视化数据点
    plt.figure(figsize=(8, 6))
    plt.scatter(C1_data[:, 0], C1_data[:, 1], color='blue', label='C1')
    plt.scatter(C2_data[:, 0], C2_data[:, 1], color='red', label='C2')
    plt.scatter(O1[0], O1[1], color='green', marker='x', s=100, label='O1')
    plt.scatter(O2[0], O2[1], color='purple', marker='x', s=100, label='O2')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Data')
    plt.legend()
    plt.grid(True)
    plt.show()
liqun()