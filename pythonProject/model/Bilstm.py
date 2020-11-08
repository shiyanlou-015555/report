import torch
# 资料知识
'''
torch.nn.embedding的使用
输入：含有待提取 indices 的任意 shape 的 Long Tensor；

输出：输出 shape =（*，H），其中 * 为输入的 shape，H = embedding_dim（若输入 shape 为 N*M，则输出 shape 为 N*M*H）；
bilstm的输入
input: 输入数据，即上面例子中的一个句子（或者一个batch的句子），其维度形状为 (seq_len, batch, input_size(每个字符的映射维度))
    seq_len: 句子长度，即单词数量，这个是需要固定的。当然假如你的一个句子中只有2个单词，但是要求输入10个单词，这个时候可以用torch.nn.utils.rnn.pack_padded_sequence()或者torch.nn.utils.rnn.pack_sequence()来对句子进行填充或者截断。
    batch：就是你一次传入的句子的数量
    input_size: 每个单词向量的长度，这个必须和你前面定义的网络结构保持一致
output： 维度和输入数据类似，只不过最后的feature部分会有点不同，即 (seq_len, batch, num_directions * hidden_size)
'''
# RNN
from torch import nn
import torch.nn.functional as F
class BiRNN(torch.nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,Dropout,label_size):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab),embed_size)
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
        #self.embedding.weight.requires_grad = True

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(seq_len, 批量大小, 词向量维度)LSTM 的输入是三维，其顺序是 (seq_len, batch, input_size),
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, num_layers*hiddens)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        # 有问题encoding = torch.cat((outputs[0], outputs[-1]), -1)连接失败
        #encoding=outputs.sum(dim=0)
        #print(outputs.permute(1,2,0).shape)
        temp = outputs.permute(1,2,0)
        encoding=F.max_pool1d(temp,temp.size(2))
        #print(torch.squeeze(encoding).shape)
        temp = self.decoder(F.relu_(torch.squeeze(encoding)))
        #print(temp)
        outs = self.out(self.sig(temp))
        # print(outs[0])
        # print(outs.shape)
        #print(outs.shape)
        return outs

