import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import torch
from torch_geometric.data import Data
import variables as var
from scipy.io import loadmat
import faiss

# from transformer import Encoder

import h5py
########################################### NEGATIVE SAMPLE FUNCTIONS################################################
def negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, sample_type, proportion, epsilon):
    
    # training set negative samples  训练集负样本
    neg_train_x, neg_train_y = generate_negative_samples(train_x, sample_type, proportion, epsilon) # 训练样本 生成样本类型 概率 ε
    # validation set negative samples  验证集负样本
    neg_val_x, neg_val_y = generate_negative_samples(val_x, sample_type, proportion, epsilon)  # 验证集样本 生成样本类型 概率 ε

    # concat data 连接数据
    x = np.vstack((train_x,neg_train_x,val_x,neg_val_x,test_x))  # 按垂直方向连接
    y = np.hstack((train_y,neg_train_y,val_y,neg_val_y,test_y))  # 按水平方向连接

    # all training set  ones()用1填充，zeros()用0填充
    train_mask = np.hstack((np.ones(len(train_x)),np.ones(len(neg_train_x)),
                            np.zeros(len(val_x)),np.zeros(len(neg_val_x)),
                            np.zeros(len(test_x))))
    # all validation set
    val_mask = np.hstack((np.zeros(len(train_x)),np.zeros(len(neg_train_x)),
                          np.ones(len(val_x)),np.ones(len(neg_val_x)),
                          np.zeros(len(test_x))))
    # all test set
    test_mask = np.hstack((np.zeros(len(train_x)),np.zeros(len(neg_train_x)),
                           np.zeros(len(val_x)),np.zeros(len(neg_val_x)),
                           np.ones(len(test_x))))
    # normal training points  正常训练点
    neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(neg_train_x)),
                               np.zeros(len(val_y)), np.zeros(len(neg_val_x)),
                               np.zeros(len(test_y))))
    #transformer
    # x = np.vstack((train_x,  val_x,  test_x))  # 按垂直方向连接
    # y = np.hstack((train_y, val_y, test_y))  # 按水平方向连接
    # train_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(val_x)), np.zeros(len(test_x))))
    # val_mask = np.hstack((np.zeros(len(train_x)), np.ones(len(val_x)), np.zeros(len(test_x))))
    # test_mask = np.hstack((np.zeros(len(train_x)), np.zeros(len(val_x)), np.ones(len(test_x))))
    # neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(val_y)), np.zeros(len(test_y))))

    # find k nearest neighbours (idx) and their distances (dist) to each points in x within neighbour_mask==1  找到距离为1的K个最近邻
    dist, idx = find_neighbors(x, y, neighbor_mask, k)  # 返回值为 距离 和 索引

    return x.astype('float32'), y.astype('float32'), neighbor_mask.astype('float32'), train_mask.astype('float32'), val_mask.astype('float32'), test_mask.astype('float32'), dist, idx

# loading negative samples  生成负样本       epsilon: ε=0.1  proportion: 1, 实验中p=0.3
def generate_negative_samples(x, sample_type, proportion, epsilon):
    
    n_samples = int(proportion*(len(x)))   # 样本数
    n_dim = x.shape[-1]       # x的维数
        
    #M
    randmat = np.random.rand(n_samples,n_dim) < 0.3   # 生成一个（0,1）之间的均匀分布
    # uniform samples   均匀样本
    rand_unif = (epsilon* (1-2*np.random.rand(n_samples,n_dim)))
    #  subspace perturbation samples   子空间扰动样本
    rand_sub = np.tile(x, (proportion,1)) + randmat*(epsilon*np.random.randn(n_samples,n_dim))   # tile() 把x沿y方向复制p
    
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        neg_x = np.concatenate((rand_unif, rand_sub),0)  # 将rand_unif 和 rand_sub 进行拼接
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size = n_samples)]    # 从数组中随机抽取n个数据组成数组

    neg_y = np.ones(len(neg_x))  # 生成给定形状的新数组用1填充
    
    return neg_x.astype('float32'), neg_y.astype('float32')


################################### GRAPH FUNCTIONS ###############################################     
# find the k nearest neighbours of all x points out of the neighbour candidates    从邻居候选者中找出所有x点的k个最近邻居
def find_neighbors(x, y, neighbor_mask, k):
    
    # nearest neighbour object
    index = faiss.IndexFlatL2(x.shape[-1])  # 构建索引
    # add nearest neighbour candidates 添加最近邻候选
    index.add(x[neighbor_mask == 1])

    # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    dist_train, idx_train = index.search(x[neighbor_mask==1], k = k+1)
    # remove 1st nearest neighbours to remove self loops 去掉自循环
    dist_train, idx_train = dist_train[:,1:], idx_train[:,1:]
    # distances and idx of neighbour points for the non-neighbour candidates 非相邻候选的相邻点的距离和idx
    dist_test, idx_test = index.search(x[neighbor_mask==0], k = k)
    # concat
    dist = np.vstack((dist_train, dist_test))
    idx = np.vstack((idx_train, idx_test))
    
    return dist, idx

# create graph object out of x, y, distances and indices of neighbours   用x、y、距离和邻居的索引创建图形对象
def build_graph(x, y, dist, idx):
    
    # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)  shape(-1):读取列数
    idx_source = np.repeat(np.arange(len(x)),dist.shape[-1]).astype('int32')
    idx_source = np.expand_dims(idx_source,axis=0) # 扩展维度
    print('idx_source.size: ',idx_source.size)
    # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
    idx_target = idx.flatten()  # 压扁
    idx_target = np.expand_dims(idx_target,axis=0).astype('int32')  # 延展
    print('idx_target.size: ',idx_target.size)
    #stack source and target indices
    idx = np.vstack((idx_source, idx_target)) # 垂直连接

    # edge weights   边权重
    attr = dist.flatten()
    attr = np.sqrt(attr)
    attr = np.expand_dims(attr, axis=1)
    print(attr.size)
    # into tensors  转化为张量
    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)
    idx = torch.tensor(idx, dtype = torch.long)
    attr = torch.tensor(attr, dtype = torch.float32)
    print(x.size())
    print(y.size())
    print(idx.size())
    print(attr.size())
    #build PyTorch geometric Data object   生成PyTorch几何数据对象
    data = Data(x = x, edge_index = idx, edge_attr = attr, y = y)

    return data

########################################## DATASET FUNCTIONS ####################################   
#  
# split training data into train set and validation set   将训练数据拆分为训练集和验证集
def split_data(seed, all_train_x, all_train_y, all_test_x, all_test_y):
    np.random.seed(seed)

    val_idx = np.random.choice(np.arange(len(all_train_x)),size = int(0.15*len(all_train_x)), replace = False)
    val_mask = np.zeros(len(all_train_x))
    val_mask[val_idx] = 1
    val_x = all_train_x[val_mask == 1]; val_y = all_train_y[val_mask == 1]
    train_x = all_train_x[val_mask == 0]; train_y = all_train_y[val_mask == 0]
    # 归一化
    scaler = MinMaxScaler()
    scaler.fit(train_x[train_y == 0])
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
   
    if all_test_x is None:
        test_x = val_x
        test_y = val_y
    
    test_x = scaler.transform(all_test_x)
    test_y = all_test_y
	
    return train_x.astype('float32'), train_y.astype('float32'), val_x.astype('float32'), val_y.astype('float32'),  test_x.astype('float32'), test_y.astype('float32')


#load data
def load_dataset(dataset,seed):     
    np.random.seed(seed)    
    
    if dataset == 'MI-V':
        df = pd.read_csv("MI/experiment_01.csv")
        for i in ['02','03','11','12','13','14','15','17','18']:
            data = pd.read_csv("MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
        normal_idx = np.ones(len(df))
        for i in ['06','08','09','10']:
            data = pd.read_csv("MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)        
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.values    # data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))
        
    elif dataset == 'MI-F':
        df = pd.read_csv("MI/experiment_01.csv")
        for i in ['02','03','06','08','09','10','11','12','13','14','15','17','18']:
            data = pd.read_csv("MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
        normal_idx = np.ones(len(df))
        for i in ['04', '05', '07', '16']: 
            data = pd.read_csv("MI/experiment_%s.csv" %i)
            df = df.append(data, ignore_index = True)
            normal_idx = np.append(normal_idx,np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'],axis=1),machining_process_one_hot],axis=1)
        data = df.values
        idx = np.unique(data,axis=0, return_index = True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anomaly_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data,normal_data[test_idx]))
        test_y = np.concatenate((np.ones(len(anomaly_data)),np.zeros(len(test_idx))))
        
    elif dataset in ['OPTDIGITS', 'PENDIGITS','SHUTTLE','SATIMAGE','MAM','SATELLITE','THYROID','ANNTHYROID','MNIST']:
        if dataset == 'SHUTTLE':
            data = loadmat("shuttle.mat")
        elif dataset == 'OPTDIGITS':
            data = loadmat("optdigits.mat")
        elif dataset == 'PENDIGITS':
            data = loadmat('pendigits.mat')
        elif dataset == 'SATIMAGE':
            data = loadmat('satimage-2.mat')
        elif dataset == 'MAM':
            data = loadmat('mammography.mat')
        elif dataset == 'SATELLITE':
            data = loadmat('satellite.mat')
        elif dataset == 'THYROID':
            data = loadmat('thyroid.mat')
        elif dataset == 'ANNTHYROID':
            data = loadmat('annthyroid.mat')
        elif dataset == 'MNIST':
            data = loadmat('mnist.mat')

        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        # data = data['X'].astype(np.compat.long)  # MAM
        normal_data= data[label == 0]
        normal_label = label[label==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))
        test_y = np.concatenate((normal_label[test_idx],anom_label))
        # transformer
        # test_idx = np.random.choice(np.arange(0, len(data)), int(0.1 * len(data)), replace=False)  # 在正常数据中随机抽取数据  组成len(anom)大小的一维数组
        # train_idx = np.setdiff1d(np.arange(0, len(data)), test_idx)  # 输出在前者不在后者中的元素并去重排序
        # train_x = data[train_idx]  # x:数据，y:标签
        # train_y = label[train_idx]
        # test_x = data[test_idx]
        # test_y = label[test_idx]
        
    elif dataset in ['THYROID_1','HRSS']:
        if dataset == 'THYROID_1':
            data = pd.read_csv('annthyroid_21feat_normalised.csv').values
        if dataset == 'HRSS':
            data = pd.read_csv('HRSS.csv').values        # data = pd.read_csv('data/HRSS/HRSS.csv').to_numpy()
        label = data[:,-1].astype('float32').squeeze()  # 最后一列数据
        data = data[:,:-1].astype('float32')    # 除最后一列外所有数据
        normal_data= data[label == 0]   # 正常数据点
        normal_label = label[label==0]  # 正常标签
        anom_data = data[label == 1]    # 异常数据点
        anom_label = label[label ==1]   # 异常标签
        test_idx = np.random.choice(np.arange(0,len(normal_data)), len(anom_data), replace = False)  # 在正常数据中随机抽取数据  组成len(anom)大小的一维数组
        train_idx = np.setdiff1d(np.arange(0,len(normal_data)), test_idx)   # 输出在前者不在后者中的元素并去重排序
        train_x = normal_data[train_idx]    # x:数据，y:标签
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx],anom_data))  # 测试集数据 ：将正常数据和异常数据进行拼接
        test_y = np.concatenate((normal_label[test_idx],anom_label))    # 测试集标签 ：正常标签和异常标签
        #不用负采样
        # test_idx = np.random.choice(np.arange(0, len(data)), int(0.1 * len(data)), replace=False)  # 在正常数据中随机抽取数据  组成len(anom)大小的一维数组
        # train_idx = np.setdiff1d(np.arange(0, len(data)), test_idx)  # 输出在前者不在后者中的元素并去重排序
        # train_x = data[train_idx]  # x:数据，y:标签
        # train_y = label[train_idx]
        # test_x = data[test_idx]
        # test_y = label[test_idx]

    elif dataset == 'SMTP':
        data = h5py.File('data/SMTP/smtp.mat', 'r')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label == 0]
        anom_data = data[label == 1]
        anom_label = label[label == 1]
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anom_data), replace=False)
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx], anom_data))
        test_y = np.concatenate((normal_label[test_idx], anom_label))

    elif dataset == 'SATELLITE':
        data = loadmat('data/SATELLITE/satellite.mat')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label ==0]
        anom_data = data[label == 1]
        anom_label = label[label ==1]
        train_idx = np.random.choice(np.arange(0,len(normal_data)), 4000, replace = False)
        test_idx = np.setdiff1d(np.arange(0,len(normal_data)), train_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = normal_data[test_idx]
        test_y = normal_label[test_idx]
        test_idx = np.random.choice(np.arange(0,len(anom_data)), int(len(test_x)), replace = False)
        test_x = np.concatenate((test_x,anom_data[test_idx]))
        test_y = np.concatenate((test_y, anom_label[test_idx]))

    elif dataset in ['PEMS', 'SWaT', 'DLR', 'YH','NYC']:
        if dataset == 'PEMS':
            data = pd.read_csv('5min_2013_01_02_new.csv').values   #data = pd.read_csv('data/PeMSD7/ 5min_2013_01_02_new.csv').to_numpy()
        if dataset == 'SWaT':
            import random
            data = pd.read_csv('data/SWAT/SWaT_Dataset_del_Attack1.csv').values  #data = pd.read_csv('data/SWAT/SWaT_Dataset_del_Attack1.csv').to_numpy()
            sample_num = int(0.15 * len(data))
            sample_list = [i for i in range(len(data))]
            sample_list = random.sample(sample_list, sample_num)
            data = data[sample_list, :]
        if dataset == 'DLR':
            data = pd.read_csv('DLR.csv').values
        if dataset == 'YH':
            data = pd.read_csv('YahooA1.csv').values
        if dataset == 'NYC':
            data = pd.read_csv('NYC.csv').values
        label = data[:, -1].astype('float32').squeeze()  # 最后一列数据
        data = data[:, :-1].astype('float32')  # 除最后一列外所有数据
        normal_data = data[label == 0]  # 正常数据点
        normal_label = label[label == 0]  # 正常标签
        anom_data = data[label == 1]  # 异常数据点
        anom_label = label[label == 1]  # 异常标签
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anom_data),
                                    replace=False)  # 在正常数据中随机抽取数据  组成len(anom)大小的一维数组
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)  # 输出在前者不在后者中的元素并去重排序
        train_x = normal_data[train_idx]  # x:数据，y:标签
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx], anom_data))  # 测试集数据 ：将正常数据和异常数据进行拼接
        test_y = np.concatenate((normal_label[test_idx], anom_label))  # 测试集标签 ：正常标签和异常标签
                
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(seed, all_train_x = train_x, all_train_y = train_y, all_test_x = test_x, all_test_y = test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y
