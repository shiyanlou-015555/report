from config import config
import torchtext.vocab as pre_Vocab
config_test = config.Configurable(r'D:\PycharmProjects\pythonProject\config\db.conf')
print(config_test.pre_emb)
# print(type(config_test.pre_emb))
# cache_dir = config_test.pre_emb
# glove_vocab = pre_Vocab.GloVe(name='6B', dim=200, cache=cache_dir)
# print(glove_vocab.stoi)
print(config_test.test_dir)