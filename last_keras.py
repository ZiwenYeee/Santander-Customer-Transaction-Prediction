import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, GaussianNoise
from keras import callbacks
from scipy.stats import norm, rankdata
import keras.backend as K

import warnings


train = pd.read_csv("D:\\Kaggle\\MySecondKaggleCompetition\\train.csv")
test = pd.read_csv("D:\\Kaggle\\MySecondKaggleCompetition\\test.csv")

features = [col for col in train.columns if col not in ['target','ID_code']]
gc.collect()

test_fake = pd.read_csv("D:\\Kaggle\\MySecondKaggleCompetition\\synthetic_samples_indexes.csv")

test_fake.columns = ['ID_code']
test_fake['ID_code'] = test_fake.ID_code.apply(lambda x: "test_" + str(x))
test_fake['dis'] = 1
test = pd.merge(test, test_fake, on = ['ID_code'], how = 'left')
test.dis.fillna(0, inplace = True)
test_real = test.loc[test.dis == 0]
test_fake = test.loc[test.dis == 1]

train['flag']=1
test_real['flag']=2
test_fake['flag']=3


data=pd.concat([train,test_real]).reset_index(drop=True)
print('data.shape=',data.shape)
del train,test_real

for var in features:
    data[var]= (data[var]-data[var].mean())/data[var].std()*5


df_train=data[data['flag']==1].copy()
test_real=data[data['flag']==2].copy()
# test_fake=data[data['flag']==3].copy()
df_test=data[data['flag']>=2].copy()
df_test = pd.concat([df_test, test_fake], axis = 0)
print(df_train.shape,test_real.shape,test_fake.shape,df_test.shape)
df_train.drop(['dis'], axis = 1, inplace = True)
del data

print(len(features))


import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization, GaussianNoise
from keras import callbacks
from scipy.stats import norm, rankdata
import keras.backend as K

# LOGGER
class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=10, lr_patience=3, out_fn='', log_fn=''):
        self.auc = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
        cv_true = self.validation_data[1]
        auc_val = roc_auc_score(cv_true, cv_pred)
        if self.auc < auc_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
            self.auc = auc_val
            self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, 0.75*lr)
                print("Setting lr to {}".format(0.75*lr))
                self.no_improve_lr = 0

        return

# MODEL DEF
def _Model():
    inp = Input(shape=(600, 1))
    d1 = Dense(16, activation='relu')(inp)
    fl = Flatten()(d1)
    preds = Dense(1, activation='sigmoid')(fl)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



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

def feature_eng(train, valid, test, origin_train, origin_test,feat):
    for var in feat:
        print(var)
        data = pd.concat([origin_train[['ID_code', var]], origin_test[['ID_code', var]]])
        train['weight_' + var] = train[var].map(data.groupby([var])[var].count())
        valid['weight_' + var] = valid[var].map(data.groupby([var])[var].count())
        test['weight_'+ var] = test[var].map(data.groupby([var])[var].count())

        train['weight_' + var] = train['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * train[var]
        valid['weight_' + var] = valid['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * valid[var]
        test['weight_' + var] = test['weight_' + var].apply(lambda x: 1 if x > 1 else 0) * test[var]

    return train, valid, test


    from sklearn.model_selection import KFold, StratifiedKFold
  from sklearn.metrics import roc_auc_score

  import lightgbm as lgb

  import matplotlib.pyplot as plt
  import seaborn as sns

  import warnings

  warnings.simplefilter(action='ignore', category=FutureWarning)
  warnings.filterwarnings('ignore')

  plt.style.use('seaborn')
  sns.set(font_scale=1)
  random_state = 2
  np.random.seed(random_state)

  # df_train.reset_index(inplace = True)
  # df_test.reset_index(inplace = True)
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
  oof = df_train[['ID_code', 'target']]
  oof['predict'] = 0
  predictions = df_test[['ID_code']]
  val_aucs = []
  feature_importance_df = pd.DataFrame()


  features = [col for col in df_train.columns if col not in ['target', 'ID_code','dis','flag']]
X_test = df_test[features].values
c = 0
preds = []
c = 0
N = 1
oof_preds = np.zeros((len(df_train), 1))
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    p_valid,yp = 0,0
    X_t,y_t = augment(X_train.values, y_train.values)
    X_t = pd.DataFrame(X_t, columns = X_train.columns)
    print('after augmentation, shape = ', X_t.shape)
    X_t, X_valid, df_test = feature_eng(X_t, X_valid, df_test, df_train, test_real, features)
    print('after FE, shape = ', X_t.shape)
    feat = X_t.columns
    X_t = np.reshape(X_t.values, (-1, 600, 1))
#     y_t = y_t.values
    X_valid = np.reshape(X_valid.values, (-1, 600, 1))
    y_valid = y_valid.values
    model = _Model()
    logger = Logger(patience=10, out_path='./', out_fn='cv_{}.h5'.format(c))
    model.fit(X_t, y_t, validation_data=(X_valid, y_valid), epochs=100, verbose=2, batch_size=256,
              callbacks=[logger])
    model.load_weights('cv_{}.h5'.format(c))

    evals_result = {}
    p_valid += model.predict(X_valid)
    X_test = np.reshape(df_test[feat].values, (200000, 600, 1))
    yp += model.predict(X_test)
    c += 1
    oof['predict'][val_idx] = np.reshape(p_valid/N,(p_valid.shape[0],) )
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)


    predictions['fold{}'.format(fold+1)] = yp/N

    from time import localtime, strftime

  import pandas as pd
  import numpy as np
  from sklearn.metrics import roc_auc_score

  oof_train = oof.predict
  oof_test = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis = 1)
  score = roc_auc_score(df_train.target, oof_train)
  print(" auc score: %.5f" % int(10000*np.round(score,4)))


  output = pd.DataFrame({
      'ID_code' :df_test['ID_code'],
      'TARGET' : oof_test})
  output.to_csv('D:\Kaggle\MySecondKaggleCompetition\output/test_keras_{}_{}.csv'.format(strftime("%Y%m%d", localtime()),
                                                               int(1000*np.around(score,3))),index = None)

  output_train = pd.DataFrame({
      'ID_code' :df_train['ID_code'],
      'TARGET' : oof_train})
  output_train.to_csv('D:\Kaggle\MySecondKaggleCompetition\output/train_keras_{}_{}.csv'.format(strftime("%Y%m%d", localtime()),
                                                               int(1000*np.around(score,3))),index = None)
