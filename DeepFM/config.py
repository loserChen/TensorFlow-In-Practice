import tensorflow as tf

class Config(object):
    """
    用来存储一些配置信息
    """
    def __init__(self):
        self.feature_dict = None
        self.feature_size = None
        self.field_size = None
        self.embedding_size = 8

        self.epochs = 100
        self.deep_layers_activation = tf.nn.relu

        self.loss = "logloss"
        self.l2_reg = 0.1
        self.learning_rate = 0.1
        self.deep_layers=[32,32]


train_file = "./data/train.csv"
test_file = "./data/test.csv"

IGNORE_FEATURES = [
    'id', 'target'
]
CATEGORITAL_FEATURES = [
    'feat_cat_1', 'feat_cat_2'
]
NUMERIC_FEATURES = [
    'feat_num_1', 'feat_num_2'
]
