import pandas as pd
import time
import torch.utils.data as Data
import torch
import collections
import torchtext.vocab as Vocab
import matplotlib.pyplot as plt
import time
import torchtext.vocab as pre_Vocab
import sys,os
import torch.nn.functional as F
import Vocab
import numpy as np
from torch import nn
from config import config
# 评价函数 evaluate model
from eval.score import f1
'''
没有使用两个embedding，使用模型bilstm，虽然使用了embedding层，但是我默认选了grad=True
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'D:\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
test = pd.read_csv(config_test.test_dir)[config_test.cloums.split(',')]
# 字典创建
temp = pd.concat([train,dev,test],axis=0)
vocab_pre = Vocab.Vocab_built(max_len=10)
vocab = vocab_pre.get_vocab_comments(temp)
# 训练集合的组装
train_set = Data.TensorDataset(*vocab_pre.preprocess_comments(train, vocab))
batch_size = 128
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
# 验证集合
# dev = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')].head(1067) 验证是否收敛
dev_set = Data.TensorDataset(*vocab_pre.preprocess_comments(dev, vocab))
batch_size = 128
dev_iter = Data.DataLoader(dev_set, batch_size)# 不能打乱
# 测试集合
test_set = Data.TensorDataset(*vocab_pre.preprocess_comments(test, vocab))
batch_size = 128
test_iter = Data.DataLoader(test_set,batch_size)
# RNN数据
# 字符向量
num_epochs = 50
embed_size, num_hiddens, num_layers,dropout,label_size = int(config_test.embed_size),int(config_test.num_hiddens),int(config_test.num_layers),float(config_test.dropout),int(config_test.label_size)
lr = 0.01
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(torch.nn.init.xavier_uniform(ts), requires_grad=True)

    def _three():
        return (_one((embed_size, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    # 隐藏层系数256*256,输入层系数1027*256,偏置1*256
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_hiddens))
    b_q = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True)
    W_out = _one((num_hiddens,label_size))
    b_out = torch.nn.Parameter(torch.zeros(label_size, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q,W_out,b_out])


def init_lstm_state(batch_size, num_hiddens, device):  # 两个隐藏层
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
# 定义embedding层
embedding = nn.Embedding(len(vocab),embed_size)
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q,W_out,b_out] = params
    (H, C) = state
    inputs = embedding(inputs.permute(1,0))
    # print(inputs.shape)
    outputs = []
    # 公式的书写
    for X in inputs:
        # print(W_xi.shape)
        # print(X.shape)
        # print(H.shape)
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    temp  = torch.cat(outputs,0)
    temp = temp.resize(inputs.size(0),inputs.size(1),num_hiddens)
    # print(temp.shape)
    # print(inputs.shape)
    # temp = temp.resize(inputs.size(0)+1,inputs.size(1),num_hiddens)
    temp = temp.permute(1, 2, 0)
    #print(temp.shape)
    encode = torch.squeeze(nn.functional.max_pool1d(temp,temp.size(2)))
    #print(encode.shape)
    #print(W_out.shape)
    Y_hat = torch.matmul(encode,W_out)+b_out
    # print(Y_hat.shape)
    return Y_hat
# 验证集合函数
def pred(data_iter,net,device=None):
    if device is None:
        device = list(net.parameters())[0].device
    label = []
    label_true = []
    net = net.to(device)
    with torch.no_grad():
        net.eval()
        for X, Y in dev_iter:
            # print(torch.argmax(net(X.)), dim=1)
            # break
            # print(torch.argmax(net(X.to(device)),dim=1))
            # print(net(X.to(device)))
            # break
            label.extend(torch.argmax(net(X.to(device)), dim=1).cpu().numpy().tolist())
            label_true.extend(Y.numpy().tolist())
        net.train()
    return f1(label,label_true,classifications=2)
# 训练函数
# 预训练词向量使用
cache_dir = r"D:\glove.6B"
glove_vocab = pre_Vocab.GloVe(name='6B', dim=200, cache=cache_dir)

def load_pretrained_embedding(words, pretrained_vocab):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
    @return:
        embed: 加载到的词向量
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为len*100维度
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]# 将每个词语用训练的语言模型理解，pad,unkown
        except KeyError:
            if i!=1:
                idx = pretrained_vocab.stoi['<unk>']
                embed[i, :] = pretrained_vocab.vectors[idx]
            else:
                embed[i,:]=embed[i,:]
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    # print(embed.shape),在词典中寻找相匹配的词向量
    return embed

embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
embedding.weight.requires_grad = True # 直接加载预训练好的, 所以不需要更新它
embedding.to(device)
# def train_t(lstm,get_params,init_lstm_state,train_iter,dev_iter,device, num_epochs):
#     print("training on ", device)
#     params = get_params()
#     loss = nn.CrossEntropyLoss()#交叉熵损失函数
#     batch_count = 0
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#         for X, y ,z in train_iter:
#             state = init_lstm_state(batch_size, num_hiddens, device)
#             X = X.to(device)
#             y = y.to(device)
#             y_hat,_ = lstm(X,state,params)
#             l = loss(y_hat, y)
#             for param in params:
#                 param.grad.data.zero_()
#             l.backward()
#             for param in params:
#                 param.data -= lr*param.grad/batch_size
#             train_l_sum += l.cpu().item()
#             # print(l.cpu().item())
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             # dev_f1 = pred(dev_iter,net,device)
#             # scheduler.step(dev_f1)# 对验证集进行测试
#             n += y.shape[0]
#             batch_count += 1
#         # print('epoch %d, loss %.4f, train acc %.3f, dev_f1score %.3f,time %.1f sec'
#         #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,dev_f1,time.time() - start))
#         print('epoch %d, loss %.4f, train acc %.3f,time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,time.time() - start)
#
# train_t(lstm,get_params,init_lstm_state,train_iter,dev_iter,device,num_epochs)
# 验证集合函数
def pred(data_iter,net,state,param,device=None):
    if device is None:
        device = list(net.parameters())[0].device
    label = []
    label_true = []
    # net = net.to(device)
    with torch.no_grad():
        for X, Y,z in dev_iter:
            # print(torch.argmax(net(X.)), dim=1)
            # break
            # print(torch.argmax(net(X.to(device)),dim=1))
            # print(net(X.to(device)))
            # break
            label.extend(torch.argmax(net(X.to(device),state,param), dim=1).cpu().numpy().tolist())
            label_true.extend(Y.numpy().tolist())
    from sklearn.metrics import f1_score
    # return f1(label,label_true,classifications=2)
    return f1_score(label, label_true)
def train_t(train_iter,dev_iter,device,num_epochs):
    print("training on ", device)
    params = get_params()
    params.to(device)
    loss = nn.CrossEntropyLoss()#交叉熵损失函数
    batch_count = 0
    for epoch in range(num_epochs):
        state_c = torch.tensor([0])
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y ,z in train_iter:
            state = init_lstm_state(y.size(0), num_hiddens, device)
            X = X.to(device)
            y = y.to(device)
            y_hat = lstm(X,state,params)
            # print(y.shape)
            # print(y_hat)
            l = loss(y_hat, y)
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            for param in params:
                param.data -= lr*param.grad
            train_l_sum += l.cpu().item()
            # print(l.cpu().item())
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()

            # dev_f1 = pred(dev_iter,net,device)
            # scheduler.step(dev_f1)# 对验证集进行测试
            n += y.shape[0]
            batch_count += 1
            state_c = state
            # print(train_acc_sum / n)
        # print('epoch %d, loss %.4f, train acc %.3f, dev_f1score %.3f,time %.1f sec'
        #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,dev_f1,time.time() - start))
        print(pred(dev_iter, lstm, state_c, params, device))
        print('epoch %d, loss %.4f, train acc %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,time.time() - start))

train_t(train_iter,dev_iter,device,num_epochs)