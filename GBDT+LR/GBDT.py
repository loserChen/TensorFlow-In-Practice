import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
print('Load data...')
df_data = pd.read_csv('data/train.csv')

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]
x_all=df_data[NUMERIC_COLS]
y_all=df_data['target']

X_train,X_test,y_train,y_test = train_test_split(x_all,y_all,test_size=0.3)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',        #可选train,predict,convert_model
                            #train for training
                            #predict for predicting
    'boosting': 'gbdt',     #可选gbdt,rf,dart,goss
    'objective': 'binary',        #选择目标函数，主要有regression_l1,regression_l2,binary,multiclass
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,      #收缩比率
    'feature_fraction': 0.9,    #LightGBM 将会在每次迭代中随机选择部分特征
    'bagging_fraction': 0.8,    #类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(X_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1


y_pred = gbm.predict(X_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1


lm = LogisticRegression(penalty='l2',C=0.05,solver='sag') # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data
y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
y_pred_test_predict=lm.predict(transformed_testing_matrix)   # Give the label on each data

print(sum(y_pred_test_predict==y_test)/float(y_test.shape[0]))

NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
print("Normalized Cross Entropy " + str(NE))
