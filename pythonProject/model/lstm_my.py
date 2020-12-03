import torch
from torch import nn
import torch.nn.functional
class lstm(torch.nn.Module):
    def __init__(self,vocab,embed_size,num_hiddens,num_layers,Dropout,label_size):
        super(lstm,self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab),embed_size)
        self.num_hiddens = num_hiddens
        # self.W_xi = nn.Linear(embed_size,num_hiddens,bias=False)
        # self.W_hi = nn.Linear(num_hiddens,num_hiddens,bias=True)
        # self.W_xf = nn.Linear(embed_size,num_hiddens,bias=False)
        # self.W_hf = nn.Linear(num_hiddens,num_hiddens,bias=True)
        # self.W_xo = nn.Linear(embed_size,num_hiddens,bias=False)
        # self.W_ho = nn.Linear(num_hiddens,num_hiddens,bias=True)
        # self.W_xc = nn.Linear(embed_size,num_hiddens,bias=False)
        # self.W_hc = nn.Linear(num_hiddens,num_hiddens,bias=True)
        self.W_hp = nn.Linear(num_hiddens,num_hiddens,bias=True)
        self.W_x = nn.Linear(embed_size,4*num_hiddens,bias=False)
        self.W_h = nn.Linear(num_hiddens,4*num_hiddens,bias=True)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_hiddens, num_hiddens//2),
            nn.Dropout(p=Dropout),
            nn.Linear(num_hiddens//2,label_size)
        )
    def forward(self,inputs,device=None):
        #h需要重新定义，并且一定要可变维度
        outputs = []
        Hidden_state_arr = []
        cell_state = []
        # 公式的书写
        # print(inputs.shape)
        # print(inputs.device)
        embeddings = self.embedding(inputs.permute(1, 0))
        # print(embeddings.shape)
        temp_0 = embeddings[0]
        Hidden_state = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        Cell_state = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        # Hidden_state.to(device)
        for X in embeddings:
            # print(X.shape)
            # print(Hidden_state.shape)
            # X = X.to(device)
            # a = self.W_xi(X)
            # b = self.W_hi(Hidden_state)
            # print(a.shape)
            # print(b.shape)
            # a = self.W_xi(X)
            # print(Hidden_state.device)
            # b = self.W_hi(Hidden_state)
            # print(a+b)
            IFOC = self.W_x(X)+self.W_h(Hidden_state)
            IFO = self.sig(IFOC[:,:3*self.num_hiddens])
            I = IFO[:,:self.num_hiddens]
            F = IFO[:,self.num_hiddens:2*self.num_hiddens]
            O = IFO[:,2*self.num_hiddens:3*self.num_hiddens]
            C_tilda = self.tan(IFOC[:,3*self.num_hiddens:])
            # I = self.sig(self.W_xi(X)+self.W_hi(Hidden_state))
            # F = self.sig(self.W_xf(X)+self.W_hf(Hidden_state))
            # O = self.sig(self.W_xo(X)+self.W_ho(Hidden_state))
            # C_tilda = self.tan(self.W_xc(X)+self.W_hc(Hidden_state))
            Cell_state= F*Cell_state+I*C_tilda
            Hidden_state = O*self.tan(Cell_state)
            # Y = self.W_hp(Hidden_state)
            # outputs.append(Y)
            Hidden_state_arr.append(Hidden_state)
            cell_state.append(Cell_state)
        temp_1 = torch.cat(Hidden_state_arr,dim=0)
        temp_1 = temp_1.resize(inputs.size(1),inputs.size(0),self.num_hiddens)
        # print(temp_1.shape)
        temp_1 = temp_1.permute(1,2,0)
        temp_1 = torch.nn.functional.max_pool1d(temp_1,temp_1.size(2))
        outs = self.decoder(torch.squeeze(temp_1))
        # print(outs.shape)
        return outs
            # print(W_xi.shape)
            # print(X.shape)
            # print(H.shape)
            # I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
            # F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
            # O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
            # C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
            # C = F * C + I * C_tilda
            # H = O * C.tanh()
            # Y = torch.matmul(H, W_hq) + b_q
            # outputs.append(Y)