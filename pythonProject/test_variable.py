'''
处理可变长度的序列测试
'''
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch
seq = torch.tensor([[1,3,4], [2,0,5], [0,0,6]])
lens = [2,1,3]
packed = pack_padded_sequence(seq,lens,batch_first=False,enforce_sorted=False)
# 如果batch_first=False: 输入序列的维度是seq_len*batch_size*x，输出也是一样的
print(packed)
seq_unpacked,_ = pad_packed_sequence(packed,batch_first=False)
print(seq_unpacked)
'''
https://zhuanlan.zhihu.com/p/34418001
https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pad_packed_sequence.html
'''