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
import  model.Bilstmpad as bilstm
import Vocab
from config import config
# 评价函数 evaluate model
from eval.score import f1
'''
没有使用两个embedding，使用模型bilstm，虽然使用了embedding层，但是我默认选了grad=False,测试集和验证集已经上了，优化学习率中
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'D:\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
test = pd.read_csv(config_test.test_dir)[config_test.cloums.split(',')]
# 字典创建
temp = pd.concat([train,dev,test],axis=0)
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
# 测试集合
test_set = Data.TensorDataset(*vocab_pre.preprocess_comments(test, vocab))
batch_size = 128
test_iter = Data.DataLoader(test_set,batch_size)
# RNN数据
# 字符向量
embed_size, num_hiddens, num_layers,dropout,label_size = int(config_test.embed_size),int(config_test.num_hiddens),int(config_test.num_layers),float(config_test.dropout),int(config_test.label_size)
net = bilstm.BiRNN(vocab, embed_size, num_hiddens, num_layers,dropout,label_size)
# 验证集合函数
def pred(data_iter,net,device=None):
    if device is None:
        device = list(net.parameters())[0].device
    label = []
    label_true = []
    net = net.to(device)
    with torch.no_grad():
        net.eval()
        for X, Y,z in dev_iter:
            # print(torch.argmax(net(X.)), dim=1)
            # break
            # print(torch.argmax(net(X.to(device)),dim=1))
            # print(net(X.to(device)))
            # break
            label.extend(torch.argmax(net(X.to(device),z), dim=1).cpu().numpy().tolist())
            label_true.extend(Y.numpy().tolist())
        net.train()
    return f1(label,label_true,classifications=2)
# 训练函数

def train(train_iter,dev_iter,net, loss, optimizer, scheduler,device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y,z in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X,z)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            # print(l.cpu().item())
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            dev_f1 = pred(dev_iter,net,device)
            scheduler.step(dev_f1)
            if(dev_f1>0.763):
                print(pred(test_iter,net,device))
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, dev_f1score %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,dev_f1,time.time() - start))
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


net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

### 训练
num_epochs = 15
lr = 0.004

optimizer = torch.optim.Adam(net.parameters(),lr=lr)
# 指数调整
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.5)f1_score is :0.7577853675212161
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.5,patience=10,verbose=True,eps=0.0005)#f1_score is :0.7605179083076073
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_iter,dev_iter,net, loss, optimizer,scheduler, device, num_epochs)
# 测试
label = []
label_true = []
net = net.to(device)
for X,Y,z in test_iter:
  # print(torch.argmax(net(X.)), dim=1)
  # break
  #print(torch.argmax(net(X.to(device)),dim=1))
  #print(net(X.to(device)))
  #break
  label.extend(torch.argmax(net(X.to(device),z),dim=1).cpu().numpy().tolist())
  label_true.extend(Y.numpy().tolist())
k=0
for i,j in zip(label,label_true):
  if i==j:
    k=k+1
print(k/1060)
print('f1_score is :{}'.format(f1(label_true,label,2)))
'''
There are 23 oov words.
training on  cuda
epoch 1, loss 0.5810, train acc 0.688, dev_f1score 0.736,time 7.7 sec
epoch 2, loss 0.2105, train acc 0.804, dev_f1score 0.753,time 6.6 sec
epoch 3, loss 0.1105, train acc 0.853, dev_f1score 0.754,time 6.6 sec
0.7591377694470478
f1_score is :0.7577067180611479
training on  cuda
epoch 1, loss 0.6107, train acc 0.656, dev_f1score 0.740,time 7.4 sec
epoch 2, loss 0.2153, train acc 0.805, dev_f1score 0.751,time 6.5 sec
epoch 3, loss 0.1101, train acc 0.860, dev_f1score 0.769,time 6.5 sec
0.7575471698113208
f1_score is :0.7582622461944963
There are 28 oov words.
training on  cuda
epoch 1, loss 0.6104, train acc 0.660, dev_f1score 0.725,time 7.4 sec
epoch 2, loss 0.2173, train acc 0.797, dev_f1score 0.759,time 6.6 sec
epoch 3, loss 0.1107, train acc 0.859, dev_f1score 0.776,time 6.7 sec
epoch 4, loss 0.0655, train acc 0.891, dev_f1score 0.772,time 6.7 sec
epoch 5, loss 0.0363, train acc 0.932, dev_f1score 0.759,time 6.8 sec
epoch 6, loss 0.0201, train acc 0.958, dev_f1score 0.764,time 6.7 sec
epoch 7, loss 0.0105, train acc 0.973, dev_f1score 0.763,time 6.6 sec
epoch 8, loss 0.0070, train acc 0.983, dev_f1score 0.760,time 6.6 sec
0.7660377358490567
f1_score is :0.7656165913954696
There are 28 oov words.
training on  cuda
Epoch    17: reducing learning rate of group 0 to 2.0000e-03.
Epoch    33: reducing learning rate of group 0 to 1.0000e-03.
epoch 1, loss 0.5822, train acc 0.683, dev_f1score 0.723,time 7.8 sec
epoch 2, loss 0.2421, train acc 0.764, dev_f1score 0.738,time 6.8 sec
epoch 3, loss 0.1502, train acc 0.787, dev_f1score 0.729,time 6.7 sec
epoch 4, loss 0.1058, train acc 0.802, dev_f1score 0.733,time 7.0 sec
epoch 5, loss 0.0794, train acc 0.819, dev_f1score 0.732,time 6.9 sec
epoch 6, loss 0.0609, train acc 0.834, dev_f1score 0.745,time 6.9 sec
epoch 7, loss 0.0468, train acc 0.857, dev_f1score 0.744,time 7.1 sec
0.7603773584905661
f1_score is :0.7605179083076073
'''