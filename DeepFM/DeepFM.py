import numpy as np
import pandas as pd
import tensorflow as tf
from DeepFM.DataReader import FeatureDictionary
from DeepFM.DataReader import parse
import DeepFM.config as con

##################################
# 1. 配置信息
##################################

config = con.Config()

##################################
# 2. 读取文件
##################################
dfTrain = pd.read_csv(con.train_file)
dfTest = pd.read_csv(con.test_file)


##################################
# 3. 准备数据
##################################

# FeatureDict
config.feature_dict, config.feature_size = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest, numeric_cols=con.NUMERIC_FEATURES, ignore_cols=con.IGNORE_FEATURES)
print(config.feature_dict)
print(config.feature_size)
# Xi, Xv
Xi_train, Xv_train, y = parse(feat_dict=config.feature_dict, df=dfTrain, has_label=True)
Xi_test, Xv_test, ids = parse(feat_dict=config.feature_dict, df=dfTest, has_label=False)
config.field_size = len(Xi_train[0])
print(Xi_train)
print(Xv_train)
print(config.field_size)

##################################
# 4. 建立模型
##################################

# 模型参数

# BUILD THE WHOLE MODEL
tf.set_random_seed(2018)


# init_weight
weights = dict()
# Sparse Features 到 Dense Embedding的全连接权重。[其实是Embedding]
weights['feature_embedding'] = tf.Variable(initial_value=tf.random_normal(shape=[config.feature_size, config.embedding_size],mean=0,stddev=0.1),
                                           name='feature_embedding',
                                           dtype=tf.float32)
# Sparse Featues 到 FM Layer中Addition Unit的全连接。 [其实是Embedding,嵌入后维度为1]
weights['feature_bias'] = tf.Variable(initial_value=tf.random_uniform(shape=[config.feature_size, 1],minval=0.0,maxval=1.0),
                                      name='feature_bias',
                                      dtype=tf.float32)
# Hidden Layer
num_layer = len(config.deep_layers)
input_size = config.field_size * config.embedding_size
glorot = np.sqrt(2.0 / (input_size + config.deep_layers[0])) # glorot_normal: stddev = sqrt(2/(fan_in + fan_out))
weights['layer_0'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size, config.deep_layers[0]],mean=0,stddev=glorot),
                                 dtype=tf.float32)
weights['bias_0'] = tf.Variable(initial_value=tf.random_normal(shape=[1, config.deep_layers[0]],mean=0,stddev=glorot),
                                dtype=tf.float32)
for i in range(1, num_layer):
    glorot = np.sqrt(2.0 / (config.deep_layers[i - 1] + config.deep_layers[i]))
    # deep_layer[i-1] * deep_layer[i]
    weights['layer_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[config.deep_layers[i - 1], config.deep_layers[i]],mean=0,stddev=glorot),
                                          dtype=tf.float32)
    # 1 * deep_layer[i]
    weights['bias_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[1, config.deep_layers[i]],mean=0,stddev=glorot),
                                         dtype=tf.float32)
# Output Layer
deep_size = config.deep_layers[-1]
fm_size = config.field_size + config.embedding_size
input_size = fm_size + deep_size
glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size,1],mean=0,stddev=glorot),
                                           dtype=tf.float32)
weights['concat_bias'] = tf.Variable(tf.constant(value=0.01), dtype=tf.float32)


# build_network
feat_index = tf.placeholder(dtype=tf.int32, shape=[None, config.field_size], name='feat_index') # [None, field_size]
feat_value = tf.placeholder(dtype=tf.float32, shape=[None, config.field_size], name='feat_value') # [None, field_size]
label = tf.placeholder(dtype=tf.float16, shape=[None,1], name='label')

# Sparse Features -> Dense Embedding
embeddings_origin = tf.nn.embedding_lookup(weights['feature_embedding'], ids=feat_index) # [None, field_size, embedding_size]

feat_value_reshape = tf.reshape(tensor=feat_value, shape=[-1, config.field_size, 1]) # -1 * field_size * 1

# --------- 一维特征 -----------
y_first_order = tf.nn.embedding_lookup(weights['feature_bias'], ids=feat_index) # [None, field_size, 1]
w_mul_x = tf.multiply(y_first_order, feat_value_reshape) # [None, field_size, 1]  Wi * Xi
y_first_order = tf.reduce_sum(input_tensor=w_mul_x, axis=2) # [None, field_size]

# --------- 二维组合特征 ----------
embeddings = tf.multiply(embeddings_origin, feat_value_reshape) # [None, field_size, embedding_size] multiply不是矩阵相乘，而是矩阵对应位置相乘。这里应用了broadcast机制。

# sum_square part 先sum，再square
summed_features_emb = tf.reduce_sum(input_tensor=embeddings, axis=1) # [None, embedding_size]
summed_features_emb_square = tf.square(summed_features_emb)

# square_sum part
squared_features_emb = tf.square(embeddings)
squared_features_emb_summed = tf.reduce_sum(input_tensor=squared_features_emb, axis=1) # [None, embedding_size]

# second order
y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_features_emb_summed)


# ----------- Deep Component ------------
y_deep = tf.reshape(embeddings, shape=[-1, config.field_size * config.embedding_size]) # [None, field_size * embedding_size]
for i in range(0, len(config.deep_layers)):
    y_deep = tf.add(tf.matmul(y_deep, weights['layer_%d' % i]), weights['bias_%d' % i])
    y_deep = config.deep_layers_activation(y_deep)

# ----------- output -----------
concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
out = tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias'])
out = tf.nn.sigmoid(out)

config.loss = "logloss"
config.l2_reg = 0.1
config.learning_rate = 0.1

# loss
if config.loss == "logloss":
    loss = tf.losses.log_loss(label, out)
elif config.loss == "mse":
    loss = tf.losses.mean_squared_error(label, out)

# l2
if config.l2_reg > 0:
    loss += tf.contrib.layers.l2_regularizer(config.l2_reg)(weights['concat_projection'])
    for i in range(len(config.deep_layers)):
        loss += tf.contrib.layers.l2_regularizer(config.l2_reg)(weights['layer_%d' % i])

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

##################################
# 5. 训练
##################################

# init session
sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())

# train
feed_dict = {
    feat_index: Xi_train,
    feat_value: Xv_train,
    label:      np.array(y).reshape((-1,1))
}


for epoch in range(config.epochs):
    train_loss,opt = sess.run((loss, optimizer), feed_dict=feed_dict)
    print("epoch: {0}, train loss: {1:.6f}".format(epoch, train_loss))




##################################
# 6. 预测
##################################
dummy_y = [1] * len(Xi_test)
feed_dict_test = {
    feat_index: Xi_test,
    feat_value: Xv_test,
    label: np.array(dummy_y).reshape((-1,1))
}

prediction = sess.run(out, feed_dict=feed_dict_test)

sub = pd.DataFrame({"id":ids, "pred":np.squeeze(prediction)})
print("prediction:")
print(sub)
