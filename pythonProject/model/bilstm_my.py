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
        self.W_xf = nn.Linear(embed_size,4*num_hiddens,bias=False)
        self.W_hf = nn.Linear(num_hiddens,4*num_hiddens,bias=True)
        self.W_xb = nn.Linear(embed_size,4*num_hiddens,bias=False)
        self.W_hb = nn.Linear(num_hiddens,4*num_hiddens,bias=True)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(2*num_hiddens, num_hiddens//2),
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
        Hidden_state_f = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        Cell_state_f = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        Hidden_state_b = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        Cell_state_b = torch.zeros(temp_0.size(0),self.num_hiddens,device=device,requires_grad=True)# 初始化一个需要中间使用的变量
        # Hidden_state.to(device)
        embeddings_1 = embeddings.__reversed__()
        for X,X_b in zip(embeddings,embeddings_1):
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
            # 前向
            IFOC_f = self.W_xf(X)+self.W_hf(Hidden_state_f)
            IFO_f = self.sig(IFOC_f[:,:3*self.num_hiddens])
            I_f = IFO_f[:,:self.num_hiddens]
            F_f = IFO_f[:,self.num_hiddens:2*self.num_hiddens]
            O_f = IFO_f[:,2*self.num_hiddens:3*self.num_hiddens]
            C_tilda_f = self.tan(IFOC_f[:,3*self.num_hiddens:])
            Cell_state_f= F_f*Cell_state_f+I_f*C_tilda_f
            Hidden_state_f = O_f*self.tan(Cell_state_f)
            # 反向
            IFOC_b = self.W_xb(X_b)+self.W_hb(Hidden_state_b)
            IFO_b = self.sig(IFOC_b[:,:3*self.num_hiddens])
            I_b = self.sig(IFO_b[:,:self.num_hiddens])
            F_b = self.sig(IFO_b[:,self.num_hiddens:2*self.num_hiddens])
            O_b = self.sig(IFO_b[:,2*self.num_hiddens:3*self.num_hiddens])
            C_tilda_b = self.tan(IFOC_b[:,3*self.num_hiddens:])
            Cell_state_b= F_b*Cell_state_b+I_b*C_tilda_b
            Hidden_state_b = O_b*self.tan(Cell_state_b)
            temp_2 = torch.cat((Hidden_state_f,Hidden_state_b),dim=1)
            # print(temp_2.shape)
            # Y = self.W_hp(Hidden_state)
            # outputs.append(Y)
            Hidden_state_arr.append(temp_2)
            # cell_state.append(Cell_state)
        temp_1 = torch.cat(Hidden_state_arr,dim=0)
        temp_1 = temp_1.resize(inputs.size(1),inputs.size(0),2*self.num_hiddens)
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
            # F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)`
            # O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
            # C_tilda = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
            # C = F * C + I * C_tilda
            # H = O * C.tanh()
            # Y = torch.matmul(H, W_hq) + b_q
            # outputs.append(Y)