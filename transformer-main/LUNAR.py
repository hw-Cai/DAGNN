# imports
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import utils
import variables as var
from torch_geometric.nn import MessagePassing
from copy import deepcopy

from transformer import Encoder
from transformer import Transformer
from torch_geometric.nn import GCNConv
# Message passing scheme    GNN消息传递框架
class GNN1(MessagePassing):
    def __init__(self,k):
        super(GNN1, self).__init__(flow="target_to_source")  # flow定义消息传递的流向  这里指(j,i) ∈ E 是边的集合
        self.k = k
        self.hidden_size = 256

        self.network = nn.Sequential(

            #GNN
            nn.Linear(k,self.hidden_size),    # 隐藏层
            nn.Tanh(),

            #GCN
            # nn.Conv1d(k, self.hidden_size, kernel_size=3, padding=1),  # 一维卷积层
            # nn.Tanh()

            # # nn.ReLU()
            # nn.ELU()
            # nn.Linear(self.hidden_size,self.hidden_size),  # 隐藏层
            # nn.Tanh(),
            # nn.Linear(self.hidden_size,self.hidden_size),   # 隐藏层
            # nn.Tanh(),
            # # ###############Encoder-Decoder将下两层去掉######
            # nn.Linear(self.hidden_size,1),    # 输出层
            # nn.Sigmoid()    # 输出范围[0,1], tanh在特征相差明显时的效果会很好 ,在二分类问题中，一般tanh被用在隐层，而sigmoid被用在输出层。
            )

     # forward函数的作用是将输入数据经过网络中各个层的计算和变换后，得到输出结果。   x:节点的特征矩阵  edge_index:节点间的边  edge_attr: 边权重
    def forward(self, x, edge_index, edge_attr):


        self.network = self.network.to(dtype = torch.float32)    # 转化为float32 的Tensor数据类型

        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, k=self.k, network=self.network)
        return out


    def message(self,x_i,x_j,edge_attr):
        # message is the edge weight  ...message是边的权重
        #print('------------message--------')
        return edge_attr

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages    连接全部的k条消息

        self.input_aggr = inputs.reshape(-1,k)    # reshape(行,列) 转化为k列的矩阵 默认为100
        # pass through network  通过网络
        #print('---------network--------')
        out = self.network(self.input_aggr)
        return out



'''

将GNN改为Encoder-decoder  

'''

# GNN
class GNN(torch.nn.Module):
    def __init__(self, k):
        super(GNN, self).__init__()
        self.k = k
        self.L1 = GNN1(self.k)

        ###加入代码###
        self.encoder = nn.Sequential(
            # ########修改的encoder 结构########
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.ReLU(),
            # nn.ELU(),
            nn.Linear(128, 6),
            nn.Tanh()
            # nn.ReLU()
            # nn.ELU()


            # only autoencoder  dim_of_data - 256 - 128 - 6
            # nn.Linear(9, 256),
            # nn.Tanh(),
            # nn.Linear(256, 128),
            # nn.Tanh(),
            # nn.Linear(128, 6),
            # nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            # # nn.ReLU(),
            # nn.ELU(),
            nn.Linear(128, 256),


            # only autoencoder  6 - 128 - 256 - dim_of_data
            # nn.Linear(6, 128),
            # nn.Tanh(),
            # nn.Linear(128, 256),
            # nn.Tanh(),
            # nn.Linear(256, 9),
            # nn.Tanh(),
        )

        # PEMS选择(6,16), NYC选择(3,1)
        # self.GCN1 = GCNConv(6, 16)
        # self.GCN2 = GCNConv(16, 16)
        # self.GCN3 = GCNConv(16, 16)
        self.GCN1 = GCNConv(3, 1)
        # self.GCN2 = GCNConv(1, 1)
        # self.GCN3 = GCNConv(1, 1)
        self.GCN4 = GCNConv(256, 128)


        # transformer-encoder
        # self.trans = Encoder()
        # self.transf = Transformer()

        #output
        self.output = nn.Sequential(
            nn.Tanh(),
            # # nn.ReLU(),
            # nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, num_layers=1)  #hidden_size=64

    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x

        # lstm
        lstm_output, _ = self.lstm(self.x.unsqueeze(1))
        lstm_output = lstm_output.squeeze(1)
        out = self.L1(lstm_output, self.edge_index, self.edge_attr)

        # out = self.GCN1(lstm_output, self.edge_index)
        # self.x = torch.tanh(self.x)

        out = self.GCN4(out, self.edge_index)
        # out = self.GCN1(self.x, self.edge_index)
        out = self.L1(out, self.edge_index, self.edge_attr)

        # 原始GNN
        # out = self.L1(self.x, self.edge_index, self.edge_attr)

        # out = self.GCN(out, self.edge_index)
        print('out.shape: ',out.shape)

        #####添加代码 Encoder-Decoder#####
        # print('------------L2------------')
        out1 = self.encoder(out)
        decoder = self.decoder(out1)
        decoder = self.output(decoder)
        decoder = torch.squeeze(decoder, 1)
        return decoder

        # LUNAR
        # out = torch.squeeze(out, 1)  # 降维，将输入张量形状中的1去除
        # return out


def run(train_x,train_y,val_x,val_y,test_x,test_y,dataset,seed,k,samples,train_new_model):  

    # loss function   均方误差  按原始维度输出
    criterion = nn.MSELoss(reduction = 'none')    

    # path to save model parameters   模型参数的保存路径
    model_path = 'saved_models/%s/%d/net_%d.pth' %(dataset,k,seed)
    # model_path = 'saved_models/%s/%d/02_net_%d.pth' % (dataset, k, seed)    #添加的代码
    if not os.path.exists(os.path.dirname(model_path)):  # 判断路径中的文件是否存在
       os.makedirs(os.path.dirname(model_path))   # 创建目录
    
    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon)
    data = utils.build_graph(x, y, dist, idx)
    # print(data)
    print('data_size: ',data.size())

    #用transformer重新编码
    # data = np.array(data)
    # data = data.astype('float')
    # data = torch.tensor(data)
    # encoder = Encoder().to(var.device)
    # outputs, attention = encoder(data)
    # data = outputs

    data = data.to(var.device)  # 模型和数据放到cpu或gpu上
    torch.manual_seed(seed)     # 设置cpu生成随机数的种子，方便复现
    #net, decoder = GNN(k).to(var.device)
    net = GNN(k).to(var.device)
    print(net)
    if train_new_model == True:
        # Adma算法 net.parameters:可用于迭代优化的参数或者定义参数组的 lr:学习率 weight_decay:权重衰减
        optimizer = optim.Adam(net.parameters(), lr = var.lr, weight_decay = var.wd)
        # optimizer = optim.SGD(net.parameters(), lr=var.lr, momentum=0.9, weight_decay=var.wd) #太差
        # optimizer = torch.optim.Adagrad(net.parameters(), lr=var.lr, lr_decay=0, weight_decay=var.wd,initial_accumulator_value=0, eps=1e-10)
        # with torch.no_grad(): 不影响算法本身但影响算法性能
        with torch.no_grad():

            net.eval()  # 测试模型
            out = net(data)

            # out,loss1 = net(data) # 更改LOSS
            loss = criterion(out,data.y)    #根据模型输出和真实标签表示损失
            val_loss = loss[val_mask == 1].mean()
            val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())



            best_val_score = 0
           
        # training  epoch=200
        for epoch in range(var.n_epochs):
            # print('------------epoch------------', epoch)
            net.train()
            optimizer.zero_grad()   # 将梯度归零
            out = net(data)

            # out,loss1 = net(data) # 更改LOSS

            # loss for training data only
            # loss = criterion(out[train_mask == 1],data.y[train_mask == 1]).sum()  # 损失求和
            loss = criterion(out[train_mask == 1], data.y[train_mask == 1]).sum()

            # ######更改LOSS#########
            # loss1 = loss1.sum()
            # loss=loss+loss1
            # print('loss+loss1: ',loss)  # #######

            print('epoch: ', epoch,' loss: ', loss)
            loss.backward()     # 反向传播计算得到每个参数的梯度值
            optimizer.step()    # 通过梯度下降执行下一步参数更新


            with torch.no_grad():
                net.eval()
                out = net(data)

                # out,loss1 = net(data) # 更改LOSS

                loss = criterion(out,data.y)
                #print('loss1 size : ', loss1.size(), 'loss size: ', loss.size())
                val_loss = loss[val_mask == 1].mean()
                val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())
                #print('val_loss: ', val_loss,'  val_score: ',val_score)
                # if new model gives the best validation set score
                if val_score >= best_val_score:
                          
                    # save model parameters
                    best_dict = {'epoch': epoch,
                           'model_state_dict': deepcopy(net.state_dict()),
                           'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                           'val_loss': val_loss,
                           'val_score': val_score,
                           'k': k,}
                    
                    # save best model
                    torch.save(best_dict, model_path)
                    
                    # reset best score so far
                    best_val_score = val_score
       
        # load best model
        net.load_state_dict(best_dict['model_state_dict'])

        # use encoder
        # for epoch in range(var.n_epochs):
        #     enout.train()
        #     optimizer.zero_grad()  # 将梯度归零
        #     edge_attr = data.edge_attr
        #     edge_index = data.edge_index
        #     x = data.x
        #     out = net.L1(x, edge_index, edge_attr)
        #     out = net.encoder(out)
        #     out = enout(out)
        #
        #     loss = criterion(out[train_mask == 1], data.y[train_mask == 1]).sum()
        #     print('epoch: ', epoch, ' loss: ', loss)
        #     loss.backward()  # 反向传播计算得到每个参数的梯度值
        #     optimizer.step()  # 通过梯度下降执行下一步参数更新
        #     torch.save(enout, 'output.pth')
        # enout.load('output.pth')

    # if not training a new model, load the saved model
    if train_new_model == False:
        
        load_dict = torch.load(model_path,map_location='cpu')   # 加载torch.save保存的模型文件
        net.load_state_dict(load_dict['model_state_dict'])      # 给模型对象加载训练好的模型参数
 
    # testing
    with torch.no_grad():
        net.eval()
        out=net(data)

        # #only encoder
        # edge_attr = data.edge_attr
        # edge_index = data.edge_index
        # x = data.x
        # out = net.L1(x, edge_index, edge_attr)
        # out = net.encoder(out)
        # out = enout(out)

        print(out.size())
        loss = criterion(out,data.y)
       
    # return output for test points  返回测试集结果
    return out[test_mask==1].cpu()