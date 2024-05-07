import numpy as np
import pandas as pd
import torch
from transformer import Encoder
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import time
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score


class Transf(nn.Module):
    def __init__(self):
        super(Transf, self).__init__()
        # self.enc = Encoder().cuda()
        self.transf = Transformer().cuda()
        self.L1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(36, 1),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            # ########修改的encoder 结构########
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 6),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 18),
            nn.Tanh(),
            nn.Linear(18, 36)
            # nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, X):
        out, _,_,_ = self.transf(X, X)
        print('out_size: ', out.size())
        out = torch.reshape(out, (len(out), 6 * 6))
        # out = self.encoder(out)
        # out = self.decoder(out)
        out = self.L1(out)
        # out = torch.squeeze(out, -1)
        return out


def trainsf(X, label):
    model = Transf().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    with torch.no_grad():
        model.eval()  # 测试模型
        output1 = model(X)

        # out,loss1 = net(data) # 更改LOSS
        loss = criterion(output1, label)  # 根据模型输出和真实标签表示损失

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output1 = model(X)
        print('output1_size: ', output1.size(), ' label_size: ', label.size())
        loss = criterion(output1, label).sum()
        print('Epoch:', epoch, ' loss: ', loss)
        loss.backward()
        optimizer.step()

    torch.save(model, 'model1.pth')
    print("保存模型")

def tst_transf(test_x):
    test_x = test_x.cuda()
    # print('test_x.is_cuda: ',test_x.is_cuda)
    model1 = torch.load('model1.pth')
    # print('------------- model')
    out = model1(test_x)
    # print('out.is_cuda:', out.is_cuda)
    # print(out.size())
    return out.cpu()


start_time = time.time()
# data = pd.read_csv('HRSS.csv').values
# label = data[:, -1].astype('int32').squeeze()  # 最后一列数据
# data = data[:, :-1].astype('int32')
data = loadmat('mammography.mat')
label = data['y'].astype('float32')
print('label_shape: ', label.shape)
data = data['X'].astype('int32')
avg_score = 0
count = 0
avg_RUNtime = 0
# label = torch.tensor(label)
# label = label.cuda()
# print('label_shape: ', label.shape)
for seed in [21,22,23]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    test_idx = np.random.choice(np.arange(0, len(data)), int(0.1 * len(data)),replace=False)  # 在正常数据中随机抽取数据  组成len(anom)大小的一维数组
    train_idx = np.setdiff1d(np.arange(0, len(data)), test_idx)  # 输出在前者不在后者中的元素并去重排序
    train_x = data[train_idx]    # x:数据，y:标签
    train_y = label[train_idx]
    test_x = data[test_idx]
    test_y = label[test_idx]

    X = train_x
    print('len_X: ', len(X))
    print(X)
    X = torch.tensor(X)
    X = X.cuda()
    train_y = torch.tensor(train_y)
    train_y = train_y.cuda()
    print('start_time: ', start_time)
    print('X_size: ', X.size())


    trainsf(X, train_y)

    test_x = torch.tensor(test_x)
    # test_x = test_x.cuda()
    # test_y = torch.tensor(test_y)
    out = tst_transf(test_x)
    print(out.is_cuda)
    out = out.detach().numpy()
    score = 100 * roc_auc_score(test_y, out)
    print('score: ', score)
    # encoder = Encoder().cuda()
    # outputs, attention = encoder(X)
    # print('len_outputs: ',len(outputs))
    # print('size_outputs: ',outputs.size())
    # print('len_attention: ',len(attention))
    # print('size_attention: ',attention.size())

    # lenth = len(X)
    # batchsize=200
    # index=0
    # m=int(lenth/batchsize)
    # print(m)
    # print('-----0-----')
    # dataset = X[index:index + batchsize]
    # print('dataset_size: ', dataset.size())
    # encoder = Encoder().cuda()
    # output, attention = encoder(dataset)
    # print('len_output: ', len(output))
    # print('size_output: ', output.size())
    # for i in range(1,m):
    #     index = index + batchsize
    #     print('-----', i, '-----')
    #     dataset = X[index:index+batchsize]
    #     print('dataset_size: ', dataset.size())
    #     # encoder = Encoder().cuda()
    #     outputs, attention = encoder(dataset)
    #     print('len_outputs: ',len(outputs))
    #     print('size_outputs: ',outputs.size())
    #     output = torch.cat((output, outputs))
    # print(output.size())
    end_time = time.time()
    print('end_time: ', end_time)
    time_sum = end_time - start_time
    print('TIME: ', time_sum)
    avg_score = avg_score + score
    RUNtime = end_time - start_time
    avg_RUNtime = avg_RUNtime + RUNtime
    count = count + 1
    print('-------final----------')
avg_score = avg_score / count
avg_RUNtime = avg_RUNtime / count
print('平均 Score: %.4f \t Runtime: %.4f' % (avg_score, avg_RUNtime))
