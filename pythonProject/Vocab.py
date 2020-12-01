import torchtext.vocab as Vocab
import collections
import torch
class Vocab_built(object):
    '''
    Vocab的创建
    '''
    def __init__(self,max_len):
        self.max_len = max_len
    # 词典构建
    def get_tokenized_comments(self,data):
        """
        data: list of [string, label]
        """

        def tokenizer(text):
            # print(jieba(text))
            return text.split()# 这是针对英文进行处理

        return [tokenizer(comment) for comment in data['data']]
# a b c a:1 b:2 c:3 abb  a00  3
# a b c : 1 2 3
    def get_vocab_comments(self,data):
        tokenized_data = self.get_tokenized_comments(data)
        counter = collections.Counter([tk for st in tokenized_data for tk in st])
        return Vocab.Vocab(counter, min_freq=3)

    def preprocess_comments(self,data, vocab):
        """因为每条评论长度不一致所以不能直接组合成小批量，我们定义preprocess_imdb函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成50。"""

        # 将每条评论通过截断或者补0，使得长度变成150
        def pad(x):
            return x[:self.max_len] if len(x) > self.max_len else x + [1] * (self.max_len - len(x))
# '<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4
        tokenized_data = self.get_tokenized_comments(data)
        features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
        # print(features)
        labels = torch.tensor([score for score in data['target']])  # 需要优化
        seq_len = torch.tensor([seq for seq in data['seq_len']])
        return features, labels,seq_len