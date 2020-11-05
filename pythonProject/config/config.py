from configparser import ConfigParser
import sys
sys.path.append('.')
class Configurable(object):
    def __init__(self,config_file):
        config = ConfigParser()
        config.read(config_file,encoding='utf8')
        self.__config = config
    @property
    def train_dir(self):
        return self.__config.get('train','train_file')
    @property
    def dev_dir(self):
        return self.__config.get('train','dev_file')
    @property
    def cloums(self):
        return self.__config.get('train','cloums')
    @property
    def embed_size(self):
        return self.__config.get('train','embed_size')
    @property
    def num_hiddens(self):
        return self.__config.get('train','num_hiddens')
    @property
    def num_layers(self):
        return self.__config.get('train','num_layers')
    @property
    def dropout(self):
        return self.__config.get('train','dropout')
    @property
    def label_size(self):
        return self.__config.get('train','label_size')