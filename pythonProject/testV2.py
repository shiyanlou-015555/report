import pandas as pd
import time
import torch.utils.data as Data
import torch
import time
import torchtext.vocab as pre_Vocab
import  model.Bilstm1pad as  bilstm
import Vocab
from config import config
from eval.score import f1
import random
'''
更改词表信息，使用了连个embedding的数据，padding数据归0，使用模型bilstm1
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'D:\PycharmProjects\pythonProject\config\db.conf')
seed_num = int(config_test.seed)
random.seed(seed_num)
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
embed_size, num_hiddens, num_layers,dropout,label_size= int(config_test.embed_size),int(config_test.num_hiddens),int(config_test.num_layers),float(config_test.dropout),int(config_test.label_size)
net = bilstm.BiRNN(vocab, embed_size, num_hiddens, num_layers,dropout,label_size,seed_num)
# 初始化
net.embedding2.weight.requires_grad = True #需要更新embedding2
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
    from sklearn.metrics import f1_score
    # return f1(label,label_true,classifications=2)
    return f1_score(label, label_true)
# 训练函数
model_num = 1
def train(train_iter,dev_iter,net, loss, optimizer, scheduler,device, num_epochs,model_num):
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
            if(dev_f1>0.780):
                if (model_num == 1):
                    print(pred(test_iter,net,device))
                    torch.save(net,r'D:\PycharmProjects\pythonProject\model_save\lstm_emb_pad.pkl')
                    model_num = model_num+1
                else:
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

net.embedding1.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding1.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它
### 训练
num_epochs = 50
lr = 0.01

optimizer = torch.optim.Adam(net.parameters(),lr=lr)
# 指数调整
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.5)f1_score is :0.7577853675212161
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max',factor=0.5,patience=10,verbose=True,eps=0.000005)#f1_score is :0.7605179083076073
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_iter,dev_iter,net, loss, optimizer,scheduler, device, num_epochs,model_num)
#测试
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
epoch 1, loss 0.6558, train acc 0.607, time 3.6 sec
epoch 2, loss 0.2534, train acc 0.755, time 1.9 sec
epoch 3, loss 0.1229, train acc 0.841, time 1.9 sec
epoch 4, loss 0.0634, train acc 0.898, time 1.9 sec
epoch 5, loss 0.0261, train acc 0.953, time 1.9 sec
epoch 6, loss 0.0125, train acc 0.975, time 1.9 sec
epoch 7, loss 0.0057, train acc 0.986, time 1.9 sec
epoch 8, loss 0.0038, train acc 0.991, time 1.9 sec
epoch 9, loss 0.0026, train acc 0.992, time 1.8 sec
0.7478912839737581
f1_socre is :0.7425837320574162
'''