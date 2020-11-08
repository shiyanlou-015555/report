import torchtext.vocab as  pre_Vocab
'''
使用预训练词向量，测试，还有很多参数
'''
# 预训练词向量使用
cache_dir = r"C:\Users\ACH\Desktop\glove.6B"
glove_vocab = pre_Vocab.GloVe(name='6B', dim=200, cache=cache_dir)
print(glove_vocab.vectors[40000])