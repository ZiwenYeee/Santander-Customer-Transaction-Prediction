import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
test_fake = pd.read_csv('synthetic_samples_indexes.csv')
test_fake.columns = ['ID_code']
test_fake['ID_code'] = test_fake.ID_code.apply(lambda x: 'test_' + str(x))
test_fake['dis'] = 1
test = pd.merge(test, test_fake, on = ['ID_code'], how = 'left')
test.dis.fillna(0, inplace=True)
test_real = test.loc[test.dis == 0]
test_fake = test.loc[test.dis == 1]

train['flag']=1
test_real['flag']=2
test_fake['flag']=3


data=pd.concat([train,test_real]).reset_index(drop=True)
print('data.shape=',data.shape)
del train,test_real

for var in features:
    data['scaled_' + var]= (data[var]-data[var].mean())/data[var].std()*5


train=data[data['flag']==1].copy()
test_real=data[data['flag']==2].copy()
# test_fake=data[data['flag']==3].copy()
test=data[data['flag']>=2].copy()
test = pd.concat([test, test_fake], axis = 0)
print(train.shape,test_real.shape,test_fake.shape,test.shape)
del data

print(len(features))


def feature_eng(train, valid, test, origin_train, origin_test,feat):
    for var in feat:
        print(var)
        data = pd.concat([origin_train[['ID_code', var]], origin_test[['ID_code', var]]])
        data['weight_' + var] = data[var].map(data.groupby([var])[var].count())
        train['weight_' + var] = train[var].map(data.groupby([var])[var].count())
        valid['weight_' + var] = valid[var].map(data.groupby([var])[var].count())
        test['weight_'+ var] = test[var].map(data.groupby([var])[var].count())
        train['binary_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * train[var]
        valid['binary_' + var] = valid['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * valid[var]
        test['binary_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * test[var]
    return train, valid, test

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


def kfold_lightgbm(x_train,x_test, feature, feature_list,test = True, params,num_folds, stratified = False):
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2)
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((num_folds, ntest))
    feature_importance_df = pd.DataFrame()
    for n_fold, (train_idx, test_idx) in enumerate(folds.split(x_train[feature],x_train['target'])):
        print('\n############################# kfold = ' + str(n_fold + 1))
        X_train, X_valid = x_train[feature].iloc[train_idx],x_train[feature].iloc[test_idx]
        y_train, y_valid = x_train['target'].iloc[train_idx],x_train['target'].iloc[test_idx]
        print('after kfold split, shape = ', X_train.shape)

        N = 1
        pred_valid, pred_test = 0,0
        for Ni in range(N):
            print('Ni = ', Ni)
            X_t, y_t = augment(X_train.values, y_train.values)
            X_t = pd.DataFrame(X_t, columns = feature)
            print('after augmentation, shape = ', X_t.shape)
            train_fe, valid_fe, test_fe = feature_eng(X_t, X_valid, x_test, x_train, test_real, feature)
            print('after FE, shape = ', train_fe.shape)
            if test:
                train_fe = train_fe
                valid_fe = valid_fe
                test_fe = test_fe
            else:
                train_fe = train_fe[feature_list]
                valid_fe = valid_fe[feature_list]
                test_fe = test_fe[feature_list]
            dtrain = lgb.Dataset(data = train_fe,
                                label = y_t,
                                free_raw_data = False, silent = True)
            dtest = lgb.Dataset(data = X_valid,
                               label = y_valid,
                               free_raw_data = False, silent = True)

            clf = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=100000,
                valid_sets=[dtrain, dtest],
                early_stopping_rounds=400,
                verbose_eval=4000
            )
            pred_valid += clf.predict(dtest.data)/N
            pred_test += clf.predict(x_test[train_fe.columns])/N

        oof_train[test_idx] = pred_valid
        oof_test_skf[n_fold,:] = pred_test
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dtest.label, oof_train[test_idx])))
        del clf, dtrain, dtest
        gc.collect()
    print("Full AUC score %.6f" % roc_auc_score(x_train['target'], oof_train))
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


params = {'metric': 'auc',

 'learning_rate': 0.01,
 'nthread': -1,
 'max_depth':1,
 'reg_lambda': 0.0,
 'objective': 'binary',

#  'colsample_bytree': 1,
 'bagging_freq': 5,
 'feature_fraction':0.05,
 'min_data_in_leaf':80,
 'min_sum_hessian_in_leaf':10,
 'boost_from_average':False,
 'tree_learner':'serial',
 'num_leaves': 13,
 'boosting_type': 'gbdt'}


features = [col for col in train.columns if col not in ['target','ID_code','flag']]
oof_train,oof_test = kfold_lightgbm(train, test,
                              features,feature_list = features, test = True, params,
                              5, stratified = False)
