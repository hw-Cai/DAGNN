import torch

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# model parameters for gnn   GNN模型参数
n_epochs = 200  # 200
lr = 0.001   # 学习率 0.001    调参后：0.002    0.005
wd = 0.1     # 权重衰减 0.1   调参后：0.01     0.001

# negative sample hyperparameters  负样本超参数
epsilon = 0.1  # 步长
proportion = 1  #1

    