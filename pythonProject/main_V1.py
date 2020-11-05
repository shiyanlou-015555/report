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
'''
没有使用两个embedding
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'C:\Users\ACH\Desktop\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
# 字典创建
temp = pd.concat([train,dev],axis=0)
vocab_pre = Vocab.Vocab_built(max_len=50)
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

net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

#####
num_epochs = 20

optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_iter,net, loss, optimizer, device, num_epochs)
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
print('f1_score is :{}'.format(f1_score(label_true,label)))
'''
There are 23 oov words.
training on  cpu
epoch 1, loss 0.6165, train acc 0.649, time 40.9 sec
epoch 2, loss 0.2519, train acc 0.750, time 39.4 sec
epoch 3, loss 0.1581, train acc 0.772, time 39.5 sec
epoch 4, loss 0.1130, train acc 0.782, time 38.0 sec
epoch 5, loss 0.0855, train acc 0.799, time 38.0 sec
epoch 6, loss 0.0678, train acc 0.813, time 39.7 sec
epoch 7, loss 0.0538, train acc 0.828, time 38.2 sec
epoch 8, loss 0.0440, train acc 0.843, time 37.5 sec
epoch 9, loss 0.0372, train acc 0.854, time 37.5 sec
epoch 10, loss 0.0305, train acc 0.870, time 37.5 sec
0.7310215557638238
f1_socre is :0.7274453941120608
There are 23 oov words.
training on  cuda
epoch 1, loss 0.6253, train acc 0.640, time 2.7 sec
epoch 2, loss 0.2498, train acc 0.753, time 1.8 sec
epoch 3, loss 0.1557, train acc 0.772, time 1.8 sec
epoch 4, loss 0.1094, train acc 0.788, time 1.7 sec
epoch 5, loss 0.0822, train acc 0.809, time 1.8 sec
epoch 6, loss 0.0651, train acc 0.819, time 1.8 sec
epoch 7, loss 0.0507, train acc 0.842, time 1.7 sec
epoch 8, loss 0.0410, train acc 0.853, time 1.7 sec
epoch 9, loss 0.0324, train acc 0.873, time 1.7 sec
epoch 10, loss 0.0255, train acc 0.888, time 1.7 sec
epoch 11, loss 0.0201, train acc 0.907, time 1.8 sec
epoch 12, loss 0.0161, train acc 0.920, time 1.8 sec
epoch 13, loss 0.0126, train acc 0.936, time 1.8 sec
epoch 14, loss 0.0098, train acc 0.947, time 1.8 sec
epoch 15, loss 0.0072, train acc 0.961, time 1.8 sec
0.7478912839737581
f1_socre is :0.7570009033423667
'''