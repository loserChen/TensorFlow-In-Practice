import tensorflow as tf

import pandas as pd
import numpy as np

import DCN.config as config

from sklearn.model_selection import StratifiedKFold
from DCN.DataLoader import FeatureDictionary, DataParser

from DCN.dcn import DCN


def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)                #每行里有数据为-1的总数，因此每行数据都有missing_feat这一个特征列
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]             #两列数据相乘
        return df

    dfTrain = preprocess(dfTrain)     #预处理后的dfTrain
    dfTest = preprocess(dfTest)       #预处理后的dfTest

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]            #删去id与target的列
    cols=  [c for c in cols if c not in config.IGNORE_COLS]                     #删去属于该忽视的列

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test

def run_base_model_dcn(dfTrain, dfTest,folds,dcn_params):

    fd = FeatureDictionary(dfTrain,dfTest,numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           cate_cols = config.CATEGORICAL_COLS)

    print(fd.feat_dim)
    print(fd.feat_dict)

    data_parser = DataParser(feat_dict=fd)
    cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    cate_Xi_test, cate_Xv_test, numeric_Xv_test, ids_test = data_parser.parse(df=dfTest)

    dcn_params["cate_feature_size"] = fd.feat_dim
    dcn_params["field_size"] = len(cate_Xi_train[0])
    dcn_params['numeric_feature_size'] = len(config.NUMERIC_COLS)

    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_, y_train_ = _get(cate_Xi_train, train_idx), _get(
            cate_Xv_train, train_idx), _get(numeric_Xv_train, train_idx), _get(y_train, train_idx)
        cate_Xi_valid_, cate_Xv_valid_, numeric_Xv_valid_, y_valid_ = _get(cate_Xi_train, valid_idx), _get(
            cate_Xv_train, valid_idx), _get(numeric_Xv_train, valid_idx), _get(y_train, valid_idx)

        dcn = DCN(**dcn_params)

        dcn.fit(cate_Xi_train_, cate_Xv_train_, numeric_Xv_train_, y_train_, cate_Xi_valid_, cate_Xv_valid_,
                numeric_Xv_valid_, y_valid_)
    dcn.predict(cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train)

dfTrain, dfTest, X_train, y_train, X_test, ids_test = load_data()
print('load_data_over')
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))
print('process_data_over')

dcn_params = {

    "embedding_size": 8,
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "random_seed": config.RANDOM_SEED,
    "cross_layer_num":3
}
print('start train')
run_base_model_dcn(dfTrain, dfTest, folds, dcn_params)