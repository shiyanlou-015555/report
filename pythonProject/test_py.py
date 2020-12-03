import torch
a = torch.tensor([[1,2,3],[2,3,4]])
print(a[:,:2])
print(a.__reversed__())

# b = torch.tensor([[1],[2]])
# print(a)
# x = torch.cat((a,b),0)
# print(x)
# print(x.view(1,2,2))
