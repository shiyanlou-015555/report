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
import  model.Bilstm as bilstm
import Vocab
from config import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'C:\Users\ACH\Desktop\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
# 字典创建
vocab_pre = Vocab.Vocab_built(max_len=50)
vocab = vocab_pre.get_vocab_comments(train)
# 训练集合的组装
train_set = Data.TensorDataset(*vocab_pre.preprocess_comments(train, vocab))
batch_size = 128
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
# RNN数据
# 字符向量
embed_size, num_hiddens, num_layers,dropout,label_size = int(config_test.embed_size),int(config_test.num_hiddens),int(config_test.num_layers),float(config_test.dropout),int(config_test.label_size)
net = bilstm.BiRNN(vocab, embed_size, num_hiddens, num_layers,dropout,label_size)
# 初始化
net.embedding.weight.requires_grad = True #需要更新embedding
# 训练函数

def train(train_iter,net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            # print(l.cpu().item())
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,time.time() - start))
# 预训练词向量使用
cache_dir = r"C:\Users\ACH\Desktop\glove.6B"
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
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    # print(embed.shape),在词典中寻找相匹配的词向量
    return embed

net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

#####
num_epochs = 10
optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_iter,net, loss, optimizer, device, num_epochs)
# 验证集合
# dev = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')].head(1067) 验证是否收敛
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
dev_set = Data.TensorDataset(*vocab_pre.preprocess_comments(dev, vocab))
batch_size = 128
dev_iter = Data.DataLoader(dev_set, batch_size)# 不能打乱
# 测试
label = []
label_true = []
net = net.to(device)
for X,Y in dev_iter:
  # print(torch.argmax(net(X.)), dim=1)
  # break
  #print(torch.argmax(net(X.to(device)),dim=1))
  #print(net(X.to(device)))
  #break
  label.extend(torch.argmax(net(X.to(device)),dim=1).cpu().numpy().tolist())
  label_true.extend(Y.numpy().tolist())
k=0
for i,j in zip(label,label_true):
  if i==j:
    k=k+1
print(k/1067)
# 评价函数 evaluate model
from sklearn.metrics import f1_score
print('f1_socre is :{}'.format(f1_score(label_true,label)))
'''
检测模型收敛性: 
training on  cpu
epoch 1, loss 0.6910, train acc 0.543, time 18.6 sec
epoch 2, loss 0.3251, train acc 0.621, time 18.5 sec
epoch 3, loss 0.1863, train acc 0.714, time 18.7 sec
epoch 4, loss 0.1134, train acc 0.784, time 18.7 sec
epoch 5, loss 0.0715, train acc 0.842, time 18.7 sec
epoch 6, loss 0.0467, train acc 0.882, time 18.8 sec
epoch 7, loss 0.0314, train acc 0.911, time 18.5 sec
epoch 8, loss 0.0201, train acc 0.938, time 18.6 sec
epoch 9, loss 0.0142, train acc 0.951, time 18.7 sec
epoch 10, loss 0.0102, train acc 0.962, time 19.4 sec
0.9615745079662605
数据测试结果：
training on  cpu: sum了output max_len = 25
epoch 1, loss 0.6787, train acc 0.568, time 10.2 sec
epoch 2, loss 0.3006, train acc 0.672, time 10.3 sec
epoch 3, loss 0.1646, train acc 0.760, time 10.9 sec
epoch 4, loss 0.0992, train acc 0.819, time 10.6 sec
epoch 5, loss 0.0619, train acc 0.864, time 10.7 sec
epoch 6, loss 0.0395, train acc 0.902, time 10.9 sec
epoch 7, loss 0.0251, train acc 0.929, time 10.4 sec
epoch 8, loss 0.0164, train acc 0.947, time 11.0 sec
epoch 9, loss 0.0114, train acc 0.958, time 12.0 sec
epoch 10, loss 0.0091, train acc 0.964, time 10.7 sec
0.6954076850984068
training on  cpu  :maxpool1d了第3个维度，max_len=25
epoch 1, loss 0.6896, train acc 0.532, time 10.1 sec
epoch 2, loss 0.3141, train acc 0.648, time 9.8 sec
epoch 3, loss 0.1661, train acc 0.757, time 9.8 sec
epoch 4, loss 0.0968, train acc 0.828, time 9.8 sec
epoch 5, loss 0.0578, train acc 0.881, time 9.8 sec
epoch 6, loss 0.0388, train acc 0.908, time 9.8 sec
epoch 7, loss 0.0222, train acc 0.942, time 10.1 sec
epoch 8, loss 0.0148, train acc 0.960, time 10.3 sec
epoch 9, loss 0.0108, train acc 0.964, time 9.9 sec
epoch 10, loss 0.0075, train acc 0.971, time 9.8 sec
epoch 11, loss 0.0058, train acc 0.978, time 11.0 sec
epoch 12, loss 0.0037, train acc 0.984, time 10.6 sec
epoch 13, loss 0.0035, train acc 0.984, time 13.1 sec
epoch 14, loss 0.0032, train acc 0.983, time 10.6 sec
epoch 15, loss 0.0025, train acc 0.987, time 9.9 sec
0.6925960637300843
training on  cpu  sum了，随后max_len=50
epoch 1, loss 0.6892, train acc 0.548, time 20.6 sec
epoch 2, loss 0.3165, train acc 0.637, time 18.4 sec
epoch 3, loss 0.1776, train acc 0.737, time 18.2 sec
epoch 4, loss 0.1065, train acc 0.807, time 18.4 sec
epoch 5, loss 0.0666, train acc 0.854, time 18.3 sec
epoch 6, loss 0.0427, train acc 0.896, time 18.2 sec
epoch 7, loss 0.0286, train acc 0.922, time 18.2 sec
epoch 8, loss 0.0191, train acc 0.941, time 18.9 sec
epoch 9, loss 0.0138, train acc 0.951, time 19.6 sec
epoch 10, loss 0.0099, train acc 0.964, time 21.0 sec
epoch 11, loss 0.0077, train acc 0.972, time 18.7 sec
epoch 12, loss 0.0059, train acc 0.975, time 18.4 sec
epoch 13, loss 0.0043, train acc 0.980, time 18.8 sec
epoch 14, loss 0.0033, train acc 0.983, time 19.9 sec
epoch 15, loss 0.0033, train acc 0.982, time 18.7 sec
0.7047797563261481
采用预训练模型: 结果sum类型和maxpool1d类型
There are 20 oov words.
training on  cpu
epoch 1, loss 0.6846, train acc 0.564, time 32.2 sec
epoch 2, loss 0.2828, train acc 0.706, time 32.6 sec
epoch 3, loss 0.1739, train acc 0.732, time 32.5 sec
epoch 4, loss 0.1248, train acc 0.748, time 31.3 sec
epoch 5, loss 0.0940, train acc 0.766, time 31.1 sec
epoch 6, loss 0.0759, train acc 0.783, time 32.4 sec
epoch 7, loss 0.0620, train acc 0.789, time 31.2 sec
epoch 8, loss 0.0525, train acc 0.802, time 30.5 sec
epoch 9, loss 0.0437, train acc 0.809, time 30.5 sec
epoch 10, loss 0.0371, train acc 0.828, time 30.5 sec
0.7282099343955014 F1_score = 0.733

'''