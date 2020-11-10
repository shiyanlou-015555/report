import torch
# 资料知识
'''
改进: 两个embedding层使用,第一个embedding层不更新(采用预训练词向量)，第二个embedding层更新
'''
# RNN
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn.functional as F
import random
class BiRNN(torch.nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,Dropout,label_size,seed_num):
        super(BiRNN, self).__init__()
        #随机种子
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        #采用两个embedding层
        self.embedding1 = nn.Embedding(len(vocab),embed_size)
        self.embedding2 = nn.Embedding(len(vocab),embed_size)
        # a b c : 1 2 3 1*3 1: [0.01,0.02,2.96]  词表长度*dim
        # bidirectional设为True即得到双向循环神经网络
        # 8000/128 or 64 sgd 128*50  128*50*100
        self.encoder = nn.LSTM(input_size=embed_size,
                    hidden_size=num_hiddens,
                    num_layers=num_layers,
                    batch_first=False,
                    dropout = Dropout,bidirectional=True
                    )# 输入 seq_len*batch_size*
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.dropout = nn.Dropout(p=Dropout)
        self.decoder = nn.Linear(num_layers*num_hiddens, num_hiddens//2)
        self.sig = nn.ReLU()
        self.out = nn.Linear(num_hiddens//2,label_size)# target_size
        nn.init.xavier_uniform(self.embedding2.weight)
        #self.embedding.weight.requires_grad = True

    def forward(self, inputs,list1):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(seq_len, 批量大小, 词向量维度)LSTM 的输入是三维，其顺序是 (seq_len, batch, input_size),
        # 改进
        embedding1 = self.embedding1(inputs.permute(1, 0))
        embedding2 = self.embedding2(inputs.permute(1, 0))
        embeddings = torch.add(embedding1,embedding2)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, num_layers*hiddens)
        embed_pad = pack_padded_sequence(embeddings,list1,batch_first=False,enforce_sorted=False)
        outputs, _ = self.encoder(embed_pad) # output, (h, c)
        embed_pack,_ = pad_packed_sequence(outputs,batch_first=False)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        # 有问题encoding = torch.cat((outputs[0], outputs[-1]), -1)连接失败
        #encoding=outputs.sum(dim=0)
        #print(outputs.permute(1,2,0).shape)
        temp = embed_pack.permute(1,2,0)
        encoding=F.max_pool1d(temp,temp.size(2))
        #print(torch.squeeze(encoding).shape)
        temp = self.decoder(F.relu_(torch.squeeze(encoding)))
        #print(temp)
        outs = self.out(self.sig(temp))
        return outs

