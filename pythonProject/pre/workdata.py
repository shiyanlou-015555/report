import pandas as pd
import nltk# punkt文件是分词器
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))# english stop word remove
import re
# 自己写入数据
data = pd.read_table('mr.test.txt',header=None,encoding='utf8')[0].str.split(' ',n=2,expand=True).iloc[:,[0,2]]#target |||| data
data.columns = ['target','data']
data['data'] = data['data'].apply(lambda x:x.lower())
data['target'] = data['target'].astype('int') #不转类型
data['data'] = data['data'].astype('string')
#采用nltk
data['data'] = data['data'].apply(lambda x:' '.join([i  for  i in nltk.word_tokenize(x) if i not in stop_words]))
data['data'] = data['data'].apply(lambda x:' '.join(re.findall('[a-zA-Z0-9\']+', x, re.S)))
data['data'] = data['data'].apply(lambda x:x.strip())
data.to_csv('test_white.csv')