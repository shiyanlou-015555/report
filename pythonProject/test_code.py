import pandas as pd
import torch.utils.data as Data
import torch
import time
import torchtext.vocab as pre_Vocab
import  model.Bilstmpad as bilstm
import Vocab
from config import config
'''
没有使用两个embedding，使用模型bilstm，虽然使用了embedding层，但是我默认选了grad=True
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_test = config.Configurable(r'D:\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
#print(train.head())
'''
   target                                               data  seq_len
0       1  brings irresistible blend warmth humor consist...       12
1       0  skirmishes power waged among victims predators...       11
2       0  post september 11 sum fears seems tacky repreh...       18
3       1  unique stubborn charismatic want better succes...        6
4       1  lightweight filmmaking sure 's pleasant enough...        9
'''
# 字典创建
temp = pd.concat([train,dev],axis=0)
vocab_pre = Vocab.Vocab_built(max_len=50)
vocab = vocab_pre.get_vocab_comments(temp)
# 训练集合的组装
train_set = Data.TensorDataset(*vocab_pre.preprocess_comments(train, vocab))
batch_size = 128
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
'''
for x,y,z in train_iter:
    print(z.numpy().tolist())
    break
'''
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
            n += y.shape[0]
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,time.time() - start))
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
net.embedding.weight.requires_grad = True # 直接加载预训练好的, 所以不需要更新它

#####
num_epochs = 10

optimizer = torch.optim.Adam(net.parameters())
loss = torch.nn.CrossEntropyLoss()# softmax,交叉熵
train(train_iter,net, loss, optimizer, device, num_epochs)
# 测试
label = []
label_true = []
net = net.to(device)
for X,Y,Z in dev_iter:
  # print(torch.argmax(net(X.)), dim=1)
  # break
  #print(torch.argmax(net(X.to(device)),dim=1))
  #print(net(X.to(device)))
  #break
  label.extend(torch.argmax(net(X.to(device),Z),dim=1).cpu().numpy().tolist())
  label_true.extend(Y.numpy().tolist())
k=0
for i,j in zip(label,label_true):
  if i==j:
    k=k+1
print(k/1067)
# 评价函数 evaluate model
from eval.score import f1
print('f1_score is :{}'.format(f1(label_true,label,2)))