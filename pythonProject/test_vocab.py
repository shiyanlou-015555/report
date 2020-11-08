import Vocab
from config import config
import pandas as pd
config_test = config.Configurable(r'C:\Users\ACH\Desktop\PycharmProjects\pythonProject\config\db.conf')
# 训练集合
train = pd.read_csv(config_test.train_dir)[config_test.cloums.split(',')]
dev = pd.read_csv(config_test.dev_dir)[config_test.cloums.split(',')]
temp = pd.concat([train,dev],axis=0)
print(temp.head(10))
# 字典创建
vocab_pre = Vocab.Vocab_built(max_len=50)
vocab = vocab_pre.get_vocab_comments(train)
#print(vocab.stoi) {word:id}'<unk>': 0, '<pad>': 1, 'the': 2, 'a': 3, 'and': 4,
#print(vocab.stoi)
'''
这里需要注意，我们的pad需要补1，不能是unk
'''