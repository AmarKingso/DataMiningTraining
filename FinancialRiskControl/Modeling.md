导入所需库


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
warnings.filterwarnings('ignore')
```

减少内存中使用数据的方法


```python
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df
```

读取数据集并优化内存


```python
data_train = pd.read_csv('clean_train.csv')
data_train = reduce_mem_usage(data_train)
data_test_a = pd.read_csv('clean_test.csv')
data_test_a = reduce_mem_usage(data_test_a)
```

    Mem. usage decreased to 77.39 Mb (73.2% reduction)
    Mem. usage decreased to 19.46 Mb (72.9% reduction)
    

去除不需要的特征


```python
features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault']]
train_x = data_train[features]
test_x = data_test_a[features]
train_y = data_train['isDefault']
```

贝叶斯优化进行调参

将data_train划分为训练集和验证集


```python
bayes_trn_index, bayes_val_index = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_x, train_y))[0]
```

定义目标函数


```python
def LGB_bayesian(num_leaves,
                 max_depth,
                 max_bin,
                 bagging_fraction,
                 bagging_freq,
                 feature_fraction,
                 min_data_in_leaf,
                 min_child_weight,
                 min_split_gain,
                 min_child_samples,
                 lambda_l2):
    
    trn_x, trn_y, val_x, val_y = train_x.iloc[bayes_trn_index], train_y[bayes_trn_index],train_x.iloc[bayes_val_index], train_y[bayes_val_index]
    lgb_train = lgb.Dataset(trn_x, label=trn_y)
    lgb_valid = lgb.Dataset(val_x, label=val_y)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc', 
        'num_leaves': int(num_leaves), 
        'max_depth': int(max_depth), 
        'max_bin': int(max_bin), 
        'bagging_fraction': round(bagging_fraction, 2), 
        'bagging_freq': int(bagging_freq), 
        'feature_fraction': round(feature_fraction, 2),
        'min_data_in_leaf': int(min_data_in_leaf),
        'min_split_gain': min_split_gain, 
        'min_child_samples': int(min_child_samples), 
        'min_child_weight': min_child_weight, 
        'lambda_l2': lambda_l2, 
        'n_jobs': 8,
        'learning_rate': 0.01,
        'verbosity': -1, 
        }
    
    num_round = 10000
    model = lgb.train(params, lgb_train, num_round, valid_sets=[lgb_valid], verbose_eval=200, early_stopping_rounds=200)
    pred = model.predict(val_x, num_iteration=model.best_iteration)
    score = roc_auc_score(val_y, pred)
    
    return score
```

定义参数范围


```python
bounds_LGB = {
    'num_leaves': (30, 150), 
    'max_depth': (3, 20), 
    'max_bin': (30, 80), 
    'bagging_fraction': (0.5, 1.0), 
    'bagging_freq': (1, 50), 
    'feature_fraction': (0.5, 1.0), 
    'min_data_in_leaf':(30,150),
    'min_split_gain': (0.0, 1.0), 
    'min_child_samples': (25, 125), 
    'min_child_weight': (0.0, 10), 
    'lambda_l2': (0.0,10.0)
}
```

使用贝叶斯优化


```python
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)
init_points = 5
n_iter = 10
LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
```

    |   iter    |  target   | baggin... | baggin... | featur... | lambda_l2 |  max_bin  | max_depth | min_ch... | min_ch... | min_da... | min_sp... | num_le... |
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.719887
    [400]	valid_0's auc: 0.726103
    [600]	valid_0's auc: 0.729308
    [800]	valid_0's auc: 0.731264
    [1000]	valid_0's auc: 0.732413
    [1200]	valid_0's auc: 0.732996
    [1400]	valid_0's auc: 0.733553
    [1600]	valid_0's auc: 0.733802
    [1800]	valid_0's auc: 0.73408
    [2000]	valid_0's auc: 0.734409
    [2200]	valid_0's auc: 0.734579
    [2400]	valid_0's auc: 0.734745
    [2600]	valid_0's auc: 0.73485
    [2800]	valid_0's auc: 0.734983
    [3000]	valid_0's auc: 0.735013
    [3200]	valid_0's auc: 0.735109
    [3400]	valid_0's auc: 0.735173
    [3600]	valid_0's auc: 0.73527
    [3800]	valid_0's auc: 0.73535
    [4000]	valid_0's auc: 0.735363
    [4200]	valid_0's auc: 0.735396
    [4400]	valid_0's auc: 0.735394
    Early stopping, best iteration is:
    [4213]	valid_0's auc: 0.735405
    | [0m 1       [0m | [0m 0.7354  [0m | [0m 0.8889  [0m | [0m 12.64   [0m | [0m 0.9121  [0m | [0m 9.657   [0m | [0m 78.63   [0m | [0m 10.71   [0m | [0m 85.9    [0m | [0m 7.755   [0m | [0m 107.0   [0m | [0m 0.722   [0m | [0m 34.2    [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722221
    [400]	valid_0's auc: 0.727538
    [600]	valid_0's auc: 0.730245
    [800]	valid_0's auc: 0.731922
    [1000]	valid_0's auc: 0.732941
    [1200]	valid_0's auc: 0.733703
    [1400]	valid_0's auc: 0.734278
    [1600]	valid_0's auc: 0.734649
    [1800]	valid_0's auc: 0.734886
    [2000]	valid_0's auc: 0.735163
    [2200]	valid_0's auc: 0.735388
    [2400]	valid_0's auc: 0.735533
    [2600]	valid_0's auc: 0.735648
    [2800]	valid_0's auc: 0.735744
    [3000]	valid_0's auc: 0.735805
    [3200]	valid_0's auc: 0.735802
    Early stopping, best iteration is:
    [3073]	valid_0's auc: 0.735813
    | [95m 2       [0m | [95m 0.7358  [0m | [95m 0.6492  [0m | [95m 3.867   [0m | [95m 0.9285  [0m | [95m 3.729   [0m | [95m 63.99   [0m | [95m 7.357   [0m | [95m 59.76   [0m | [95m 0.09413 [0m | [95m 73.0    [0m | [95m 0.9491  [0m | [95m 56.15   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.72485
    [400]	valid_0's auc: 0.729387
    [600]	valid_0's auc: 0.731932
    [800]	valid_0's auc: 0.733336
    [1000]	valid_0's auc: 0.734168
    [1200]	valid_0's auc: 0.734678
    [1400]	valid_0's auc: 0.735055
    [1600]	valid_0's auc: 0.735275
    [1800]	valid_0's auc: 0.735326
    [2000]	valid_0's auc: 0.73545
    [2200]	valid_0's auc: 0.735504
    [2400]	valid_0's auc: 0.735623
    Early stopping, best iteration is:
    [2395]	valid_0's auc: 0.735628
    | [0m 3       [0m | [0m 0.7356  [0m | [0m 0.6597  [0m | [0m 45.97   [0m | [0m 0.516   [0m | [0m 0.6508  [0m | [0m 61.49   [0m | [0m 17.85   [0m | [0m 25.87   [0m | [0m 7.466   [0m | [0m 127.5   [0m | [0m 0.07572 [0m | [0m 108.8   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.72207
    [400]	valid_0's auc: 0.727488
    [600]	valid_0's auc: 0.730427
    [800]	valid_0's auc: 0.732113
    [1000]	valid_0's auc: 0.732836
    [1200]	valid_0's auc: 0.733259
    [1400]	valid_0's auc: 0.733597
    [1600]	valid_0's auc: 0.733738
    [1800]	valid_0's auc: 0.733862
    [2000]	valid_0's auc: 0.733938
    [2200]	valid_0's auc: 0.73399
    [2400]	valid_0's auc: 0.734079
    [2600]	valid_0's auc: 0.734116
    [2800]	valid_0's auc: 0.734128
    Early stopping, best iteration is:
    [2647]	valid_0's auc: 0.734195
    | [0m 4       [0m | [0m 0.7342  [0m | [0m 0.7546  [0m | [0m 24.51   [0m | [0m 0.9778  [0m | [0m 0.000120[0m | [0m 42.35   [0m | [0m 15.11   [0m | [0m 57.46   [0m | [0m 2.77    [0m | [0m 113.5   [0m | [0m 0.9186  [0m | [0m 59.34   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.715054
    [400]	valid_0's auc: 0.721969
    [600]	valid_0's auc: 0.725256
    [800]	valid_0's auc: 0.727013
    [1000]	valid_0's auc: 0.72845
    [1200]	valid_0's auc: 0.729373
    [1400]	valid_0's auc: 0.730102
    [1600]	valid_0's auc: 0.730826
    [1800]	valid_0's auc: 0.731305
    [2000]	valid_0's auc: 0.731726
    [2200]	valid_0's auc: 0.732181
    [2400]	valid_0's auc: 0.732508
    [2600]	valid_0's auc: 0.732816
    [2800]	valid_0's auc: 0.733042
    [3000]	valid_0's auc: 0.733234
    [3200]	valid_0's auc: 0.733488
    [3400]	valid_0's auc: 0.733672
    [3600]	valid_0's auc: 0.733888
    [3800]	valid_0's auc: 0.734046
    [4000]	valid_0's auc: 0.734277
    [4200]	valid_0's auc: 0.73442
    [4400]	valid_0's auc: 0.734538
    [4600]	valid_0's auc: 0.734691
    [4800]	valid_0's auc: 0.734792
    [5000]	valid_0's auc: 0.734892
    [5200]	valid_0's auc: 0.734984
    [5400]	valid_0's auc: 0.735092
    [5600]	valid_0's auc: 0.735206
    [5800]	valid_0's auc: 0.735292
    [6000]	valid_0's auc: 0.735365
    [6200]	valid_0's auc: 0.735399
    [6400]	valid_0's auc: 0.735433
    [6600]	valid_0's auc: 0.735471
    [6800]	valid_0's auc: 0.735498
    [7000]	valid_0's auc: 0.735527
    [7200]	valid_0's auc: 0.735593
    [7400]	valid_0's auc: 0.735607
    [7600]	valid_0's auc: 0.735642
    [7800]	valid_0's auc: 0.73566
    [8000]	valid_0's auc: 0.735711
    [8200]	valid_0's auc: 0.735789
    [8400]	valid_0's auc: 0.735813
    [8600]	valid_0's auc: 0.735847
    [8800]	valid_0's auc: 0.735857
    [9000]	valid_0's auc: 0.735873
    [9200]	valid_0's auc: 0.735898
    [9400]	valid_0's auc: 0.735909
    [9600]	valid_0's auc: 0.735927
    Early stopping, best iteration is:
    [9583]	valid_0's auc: 0.735934
    | [95m 5       [0m | [95m 0.7359  [0m | [95m 0.729   [0m | [95m 13.4    [0m | [95m 0.6897  [0m | [95m 6.045   [0m | [95m 68.62   [0m | [95m 4.155   [0m | [95m 93.61   [0m | [95m 5.483   [0m | [95m 46.56   [0m | [95m 0.09875 [0m | [95m 59.47   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722677
    [400]	valid_0's auc: 0.727596
    [600]	valid_0's auc: 0.730485
    [800]	valid_0's auc: 0.732313
    [1000]	valid_0's auc: 0.733366
    [1200]	valid_0's auc: 0.734026
    [1400]	valid_0's auc: 0.734513
    [1600]	valid_0's auc: 0.734802
    [1800]	valid_0's auc: 0.735032
    [2000]	valid_0's auc: 0.735199
    [2200]	valid_0's auc: 0.735344
    [2400]	valid_0's auc: 0.735485
    [2600]	valid_0's auc: 0.735557
    [2800]	valid_0's auc: 0.735641
    [3000]	valid_0's auc: 0.735728
    [3200]	valid_0's auc: 0.735784
    [3400]	valid_0's auc: 0.735824
    [3600]	valid_0's auc: 0.735883
    [3800]	valid_0's auc: 0.735925
    [4000]	valid_0's auc: 0.735955
    [4200]	valid_0's auc: 0.735964
    Early stopping, best iteration is:
    [4052]	valid_0's auc: 0.735972
    | [95m 6       [0m | [95m 0.736   [0m | [95m 0.8458  [0m | [95m 11.87   [0m | [95m 0.5972  [0m | [95m 5.56    [0m | [95m 58.92   [0m | [95m 9.312   [0m | [95m 92.01   [0m | [95m 6.397   [0m | [95m 49.99   [0m | [95m 0.3429  [0m | [95m 57.81   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722864
    [400]	valid_0's auc: 0.727909
    [600]	valid_0's auc: 0.730987
    [800]	valid_0's auc: 0.732802
    [1000]	valid_0's auc: 0.733791
    [1200]	valid_0's auc: 0.734176
    [1400]	valid_0's auc: 0.734467
    [1600]	valid_0's auc: 0.734684
    [1800]	valid_0's auc: 0.734823
    [2000]	valid_0's auc: 0.73501
    [2200]	valid_0's auc: 0.735019
    [2400]	valid_0's auc: 0.735148
    [2600]	valid_0's auc: 0.73523
    [2800]	valid_0's auc: 0.735273
    [3000]	valid_0's auc: 0.735326
    [3200]	valid_0's auc: 0.735379
    [3400]	valid_0's auc: 0.735422
    [3600]	valid_0's auc: 0.73545
    [3800]	valid_0's auc: 0.735488
    Early stopping, best iteration is:
    [3795]	valid_0's auc: 0.73549
    | [0m 7       [0m | [0m 0.7355  [0m | [0m 0.927   [0m | [0m 11.92   [0m | [0m 0.7738  [0m | [0m 6.795   [0m | [0m 53.99   [0m | [0m 11.13   [0m | [0m 89.66   [0m | [0m 1.764   [0m | [0m 56.03   [0m | [0m 0.3906  [0m | [0m 66.24   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722007
    [400]	valid_0's auc: 0.727563
    [600]	valid_0's auc: 0.730507
    [800]	valid_0's auc: 0.732423
    [1000]	valid_0's auc: 0.733219
    [1200]	valid_0's auc: 0.733832
    [1400]	valid_0's auc: 0.734116
    [1600]	valid_0's auc: 0.734368
    [1800]	valid_0's auc: 0.734496
    [2000]	valid_0's auc: 0.734591
    [2200]	valid_0's auc: 0.734714
    [2400]	valid_0's auc: 0.734757
    [2600]	valid_0's auc: 0.734822
    [2800]	valid_0's auc: 0.734851
    Early stopping, best iteration is:
    [2705]	valid_0's auc: 0.734893
    | [0m 8       [0m | [0m 0.7349  [0m | [0m 0.6992  [0m | [0m 15.03   [0m | [0m 0.9706  [0m | [0m 4.516   [0m | [0m 61.46   [0m | [0m 14.44   [0m | [0m 85.05   [0m | [0m 7.187   [0m | [0m 48.73   [0m | [0m 0.06048 [0m | [0m 56.38   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.724967
    [400]	valid_0's auc: 0.729208
    [600]	valid_0's auc: 0.731658
    [800]	valid_0's auc: 0.733148
    [1000]	valid_0's auc: 0.733965
    [1200]	valid_0's auc: 0.734433
    [1400]	valid_0's auc: 0.734693
    [1600]	valid_0's auc: 0.734874
    [1800]	valid_0's auc: 0.734957
    [2000]	valid_0's auc: 0.735027
    [2200]	valid_0's auc: 0.735062
    [2400]	valid_0's auc: 0.735131
    [2600]	valid_0's auc: 0.735066
    Early stopping, best iteration is:
    [2405]	valid_0's auc: 0.735137
    | [0m 9       [0m | [0m 0.7351  [0m | [0m 0.7482  [0m | [0m 13.33   [0m | [0m 0.5887  [0m | [0m 1.375   [0m | [0m 37.7    [0m | [0m 10.41   [0m | [0m 78.46   [0m | [0m 2.843   [0m | [0m 58.26   [0m | [0m 0.2134  [0m | [0m 123.1   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.724192
    [400]	valid_0's auc: 0.728733
    [600]	valid_0's auc: 0.731101
    [800]	valid_0's auc: 0.73274
    [1000]	valid_0's auc: 0.733381
    [1200]	valid_0's auc: 0.733869
    [1400]	valid_0's auc: 0.734168
    [1600]	valid_0's auc: 0.734536
    [1800]	valid_0's auc: 0.734636
    [2000]	valid_0's auc: 0.734718
    [2200]	valid_0's auc: 0.734692
    Early stopping, best iteration is:
    [2014]	valid_0's auc: 0.734726
    | [0m 10      [0m | [0m 0.7347  [0m | [0m 0.6853  [0m | [0m 30.35   [0m | [0m 0.8099  [0m | [0m 0.8342  [0m | [0m 50.32   [0m | [0m 8.774   [0m | [0m 28.68   [0m | [0m 4.477   [0m | [0m 62.79   [0m | [0m 0.8905  [0m | [0m 115.3   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.724212
    [400]	valid_0's auc: 0.728776
    [600]	valid_0's auc: 0.731351
    [800]	valid_0's auc: 0.732765
    [1000]	valid_0's auc: 0.733457
    [1200]	valid_0's auc: 0.733933
    [1400]	valid_0's auc: 0.734278
    [1600]	valid_0's auc: 0.734337
    [1800]	valid_0's auc: 0.734455
    [2000]	valid_0's auc: 0.734629
    [2200]	valid_0's auc: 0.734665
    [2400]	valid_0's auc: 0.734583
    Early stopping, best iteration is:
    [2214]	valid_0's auc: 0.734667
    | [0m 11      [0m | [0m 0.7347  [0m | [0m 0.5032  [0m | [0m 17.78   [0m | [0m 0.7687  [0m | [0m 0.3131  [0m | [0m 36.49   [0m | [0m 11.21   [0m | [0m 116.5   [0m | [0m 2.996   [0m | [0m 142.4   [0m | [0m 0.3347  [0m | [0m 91.1    [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.719079
    [400]	valid_0's auc: 0.724739
    [600]	valid_0's auc: 0.727585
    [800]	valid_0's auc: 0.729432
    [1000]	valid_0's auc: 0.730635
    [1200]	valid_0's auc: 0.731454
    [1400]	valid_0's auc: 0.732112
    [1600]	valid_0's auc: 0.732804
    [1800]	valid_0's auc: 0.733283
    [2000]	valid_0's auc: 0.733551
    [2200]	valid_0's auc: 0.733836
    [2400]	valid_0's auc: 0.734149
    [2600]	valid_0's auc: 0.734377
    [2800]	valid_0's auc: 0.734537
    [3000]	valid_0's auc: 0.734673
    [3200]	valid_0's auc: 0.734782
    [3400]	valid_0's auc: 0.734925
    [3600]	valid_0's auc: 0.735062
    [3800]	valid_0's auc: 0.735182
    [4000]	valid_0's auc: 0.735323
    [4200]	valid_0's auc: 0.735348
    [4400]	valid_0's auc: 0.735396
    [4600]	valid_0's auc: 0.735475
    [4800]	valid_0's auc: 0.735542
    [5000]	valid_0's auc: 0.73557
    [5200]	valid_0's auc: 0.735604
    [5400]	valid_0's auc: 0.735655
    [5600]	valid_0's auc: 0.735629
    Early stopping, best iteration is:
    [5407]	valid_0's auc: 0.735658
    | [0m 12      [0m | [0m 0.7357  [0m | [0m 0.6093  [0m | [0m 24.34   [0m | [0m 0.6911  [0m | [0m 2.69    [0m | [0m 75.27   [0m | [0m 5.334   [0m | [0m 66.15   [0m | [0m 3.799   [0m | [0m 43.06   [0m | [0m 0.6678  [0m | [0m 76.98   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.724395
    [400]	valid_0's auc: 0.729004
    [600]	valid_0's auc: 0.731413
    [800]	valid_0's auc: 0.732888
    [1000]	valid_0's auc: 0.733457
    [1200]	valid_0's auc: 0.733777
    [1400]	valid_0's auc: 0.733913
    Early stopping, best iteration is:
    [1380]	valid_0's auc: 0.733934
    | [0m 13      [0m | [0m 0.7339  [0m | [0m 0.6853  [0m | [0m 34.55   [0m | [0m 0.965   [0m | [0m 7.648   [0m | [0m 38.74   [0m | [0m 18.9    [0m | [0m 107.0   [0m | [0m 5.281   [0m | [0m 73.5    [0m | [0m 0.5638  [0m | [0m 142.9   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.723592
    [400]	valid_0's auc: 0.727924
    [600]	valid_0's auc: 0.730496
    [800]	valid_0's auc: 0.732089
    [1000]	valid_0's auc: 0.733194
    [1200]	valid_0's auc: 0.733871
    [1400]	valid_0's auc: 0.734371
    [1600]	valid_0's auc: 0.734708
    [1800]	valid_0's auc: 0.734964
    [2000]	valid_0's auc: 0.735303
    [2200]	valid_0's auc: 0.735424
    [2400]	valid_0's auc: 0.735642
    [2600]	valid_0's auc: 0.735767
    [2800]	valid_0's auc: 0.735806
    [3000]	valid_0's auc: 0.735885
    [3200]	valid_0's auc: 0.735947
    [3400]	valid_0's auc: 0.735967
    [3600]	valid_0's auc: 0.736011
    [3800]	valid_0's auc: 0.736007
    Early stopping, best iteration is:
    [3660]	valid_0's auc: 0.73604
    | [95m 14      [0m | [95m 0.736   [0m | [95m 0.7228  [0m | [95m 12.85   [0m | [95m 0.7032  [0m | [95m 7.902   [0m | [95m 74.07   [0m | [95m 7.745   [0m | [95m 62.04   [0m | [95m 8.147   [0m | [95m 81.87   [0m | [95m 0.1438  [0m | [95m 105.5   [0m |
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.723042
    [400]	valid_0's auc: 0.727529
    [600]	valid_0's auc: 0.729973
    [800]	valid_0's auc: 0.731464
    [1000]	valid_0's auc: 0.732178
    [1200]	valid_0's auc: 0.732561
    [1400]	valid_0's auc: 0.732901
    [1600]	valid_0's auc: 0.73313
    [1800]	valid_0's auc: 0.733204
    [2000]	valid_0's auc: 0.733214
    [2200]	valid_0's auc: 0.733246
    [2400]	valid_0's auc: 0.73336
    [2600]	valid_0's auc: 0.733253
    Early stopping, best iteration is:
    [2417]	valid_0's auc: 0.733369
    | [0m 15      [0m | [0m 0.7334  [0m | [0m 0.6335  [0m | [0m 37.85   [0m | [0m 0.9691  [0m | [0m 7.208   [0m | [0m 39.17   [0m | [0m 8.868   [0m | [0m 47.89   [0m | [0m 5.928   [0m | [0m 94.79   [0m | [0m 0.1459  [0m | [0m 101.3   [0m |
    =============================================================================================================================================================
    

获取最佳参数


```python
LGB_BO.max['params']
```




    {'bagging_fraction': 0.7228117783307912,
     'bagging_freq': 12.846410699432155,
     'feature_fraction': 0.7031745228966835,
     'lambda_l2': 7.902405154563362,
     'max_bin': 74.07314172395758,
     'max_depth': 7.744560295583925,
     'min_child_samples': 62.04171900452664,
     'min_child_weight': 8.146764474600342,
     'min_data_in_leaf': 81.86903052339719,
     'min_split_gain': 0.14382617412439602,
     'num_leaves': 105.49873874462759}



采用lightgbm建模，k折交叉验证来评估模型


```python
folds = 10
seed = 2020
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

train = np.zeros(train_x.shape[0])
test = np.zeros(test_x.shape[0])

for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
    print('************************************ {} ************************************'.format(str(i+1)))
    trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
    
    train_matrix = lgb.Dataset(trn_x, label=trn_y)
    valid_matrix = lgb.Dataset(val_x, label=val_y)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 
        'max_depth': int(LGB_BO.max['params']['max_depth']), 
        'max_bin': int(LGB_BO.max['params']['max_bin']), 
        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
        'bagging_freq': int(LGB_BO.max['params']['bagging_freq']), 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
        'min_split_gain': LGB_BO.max['params']['min_split_gain'], 
        'min_child_samples': int(LGB_BO.max['params']['min_child_samples']), 
        'min_child_weight': LGB_BO.max['params']['min_child_weight'], 
        'lambda_l2': LGB_BO.max['params']['lambda_l2'], 
        'learning_rate': 0.01,
        'seed': 2020,
        'nthread': 28,
        'n_jobs':24,
        'verbose': -1,
        'silent': True
    }
    
    num_round = 10000
    model = lgb.train(params, train_matrix, num_round, valid_sets=[valid_matrix], verbose_eval=200,early_stopping_rounds=200)
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_x, num_iteration=model.best_iteration)
    
    train[valid_index] = val_pred
    test = test_pred / kf.n_splits
    score = roc_auc_score(val_y, val_pred)

    print(score)
```

    ************************************ 1 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722027
    [400]	valid_0's auc: 0.726903
    [600]	valid_0's auc: 0.729765
    [800]	valid_0's auc: 0.73159
    [1000]	valid_0's auc: 0.733012
    [1200]	valid_0's auc: 0.733979
    [1400]	valid_0's auc: 0.734575
    [1600]	valid_0's auc: 0.735219
    [1800]	valid_0's auc: 0.735638
    [2000]	valid_0's auc: 0.735995
    [2200]	valid_0's auc: 0.736304
    [2400]	valid_0's auc: 0.736531
    [2600]	valid_0's auc: 0.736791
    [2800]	valid_0's auc: 0.737042
    [3000]	valid_0's auc: 0.737217
    [3200]	valid_0's auc: 0.737327
    [3400]	valid_0's auc: 0.737435
    [3600]	valid_0's auc: 0.737585
    [3800]	valid_0's auc: 0.737679
    [4000]	valid_0's auc: 0.737717
    [4200]	valid_0's auc: 0.737695
    [4400]	valid_0's auc: 0.737752
    [4600]	valid_0's auc: 0.73775
    [4800]	valid_0's auc: 0.737839
    [5000]	valid_0's auc: 0.737893
    [5200]	valid_0's auc: 0.737903
    [5400]	valid_0's auc: 0.737964
    [5600]	valid_0's auc: 0.737993
    [5800]	valid_0's auc: 0.737981
    Early stopping, best iteration is:
    [5687]	valid_0's auc: 0.738019
    0.7380186115267288
    ************************************ 2 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.72616
    [400]	valid_0's auc: 0.730651
    [600]	valid_0's auc: 0.733262
    [800]	valid_0's auc: 0.735125
    [1000]	valid_0's auc: 0.736412
    [1200]	valid_0's auc: 0.737219
    [1400]	valid_0's auc: 0.737972
    [1600]	valid_0's auc: 0.738552
    [1800]	valid_0's auc: 0.739044
    [2000]	valid_0's auc: 0.739502
    [2200]	valid_0's auc: 0.739732
    [2400]	valid_0's auc: 0.739917
    [2600]	valid_0's auc: 0.740149
    [2800]	valid_0's auc: 0.740349
    [3000]	valid_0's auc: 0.740358
    [3200]	valid_0's auc: 0.740503
    [3400]	valid_0's auc: 0.740585
    [3600]	valid_0's auc: 0.740641
    [3800]	valid_0's auc: 0.740784
    [4000]	valid_0's auc: 0.740873
    Early stopping, best iteration is:
    [3963]	valid_0's auc: 0.740886
    0.7408863922544472
    ************************************ 3 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.719692
    [400]	valid_0's auc: 0.724591
    [600]	valid_0's auc: 0.72736
    [800]	valid_0's auc: 0.729308
    [1000]	valid_0's auc: 0.73075
    [1200]	valid_0's auc: 0.731671
    [1400]	valid_0's auc: 0.732445
    [1600]	valid_0's auc: 0.732936
    [1800]	valid_0's auc: 0.733435
    [2000]	valid_0's auc: 0.733753
    [2200]	valid_0's auc: 0.734031
    [2400]	valid_0's auc: 0.734265
    [2600]	valid_0's auc: 0.734495
    [2800]	valid_0's auc: 0.734628
    [3000]	valid_0's auc: 0.734724
    [3200]	valid_0's auc: 0.734889
    [3400]	valid_0's auc: 0.734982
    [3600]	valid_0's auc: 0.735099
    [3800]	valid_0's auc: 0.735176
    [4000]	valid_0's auc: 0.735269
    [4200]	valid_0's auc: 0.735289
    [4400]	valid_0's auc: 0.735354
    Early stopping, best iteration is:
    [4369]	valid_0's auc: 0.735379
    0.7353785892779369
    ************************************ 4 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722748
    [400]	valid_0's auc: 0.72759
    [600]	valid_0's auc: 0.730314
    [800]	valid_0's auc: 0.732228
    [1000]	valid_0's auc: 0.733411
    [1200]	valid_0's auc: 0.734477
    [1400]	valid_0's auc: 0.735268
    [1600]	valid_0's auc: 0.7358
    [1800]	valid_0's auc: 0.736267
    [2000]	valid_0's auc: 0.736621
    [2200]	valid_0's auc: 0.736906
    [2400]	valid_0's auc: 0.737137
    [2600]	valid_0's auc: 0.737296
    [2800]	valid_0's auc: 0.737482
    [3000]	valid_0's auc: 0.737589
    [3200]	valid_0's auc: 0.73772
    [3400]	valid_0's auc: 0.737787
    [3600]	valid_0's auc: 0.737903
    [3800]	valid_0's auc: 0.738039
    [4000]	valid_0's auc: 0.738092
    [4200]	valid_0's auc: 0.738111
    Early stopping, best iteration is:
    [4092]	valid_0's auc: 0.738127
    0.738127352969453
    ************************************ 5 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.724256
    [400]	valid_0's auc: 0.729353
    [600]	valid_0's auc: 0.732217
    [800]	valid_0's auc: 0.7341
    [1000]	valid_0's auc: 0.735184
    [1200]	valid_0's auc: 0.73595
    [1400]	valid_0's auc: 0.736597
    [1600]	valid_0's auc: 0.737057
    [1800]	valid_0's auc: 0.737417
    [2000]	valid_0's auc: 0.737859
    [2200]	valid_0's auc: 0.738073
    [2400]	valid_0's auc: 0.738261
    [2600]	valid_0's auc: 0.738432
    [2800]	valid_0's auc: 0.738577
    [3000]	valid_0's auc: 0.738766
    [3200]	valid_0's auc: 0.738857
    [3400]	valid_0's auc: 0.738935
    [3600]	valid_0's auc: 0.739023
    [3800]	valid_0's auc: 0.73902
    Early stopping, best iteration is:
    [3630]	valid_0's auc: 0.739044
    0.7390437702049287
    ************************************ 6 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.723327
    [400]	valid_0's auc: 0.727849
    [600]	valid_0's auc: 0.730654
    [800]	valid_0's auc: 0.732644
    [1000]	valid_0's auc: 0.733646
    [1200]	valid_0's auc: 0.734566
    [1400]	valid_0's auc: 0.73515
    [1600]	valid_0's auc: 0.735545
    [1800]	valid_0's auc: 0.735992
    [2000]	valid_0's auc: 0.736364
    [2200]	valid_0's auc: 0.736572
    [2400]	valid_0's auc: 0.736761
    [2600]	valid_0's auc: 0.73694
    [2800]	valid_0's auc: 0.737108
    [3000]	valid_0's auc: 0.737135
    [3200]	valid_0's auc: 0.737215
    [3400]	valid_0's auc: 0.737256
    [3600]	valid_0's auc: 0.737316
    [3800]	valid_0's auc: 0.737357
    [4000]	valid_0's auc: 0.737411
    [4200]	valid_0's auc: 0.737419
    [4400]	valid_0's auc: 0.73743
    [4600]	valid_0's auc: 0.737474
    [4800]	valid_0's auc: 0.737542
    [5000]	valid_0's auc: 0.73751
    Early stopping, best iteration is:
    [4821]	valid_0's auc: 0.73756
    0.7375597940148356
    ************************************ 7 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722899
    [400]	valid_0's auc: 0.728011
    [600]	valid_0's auc: 0.730876
    [800]	valid_0's auc: 0.732821
    [1000]	valid_0's auc: 0.734139
    [1200]	valid_0's auc: 0.735136
    [1400]	valid_0's auc: 0.735837
    [1600]	valid_0's auc: 0.736344
    [1800]	valid_0's auc: 0.736738
    [2000]	valid_0's auc: 0.737114
    [2200]	valid_0's auc: 0.737403
    [2400]	valid_0's auc: 0.737658
    [2600]	valid_0's auc: 0.737815
    [2800]	valid_0's auc: 0.737955
    [3000]	valid_0's auc: 0.738187
    [3200]	valid_0's auc: 0.73825
    [3400]	valid_0's auc: 0.738304
    [3600]	valid_0's auc: 0.73841
    [3800]	valid_0's auc: 0.738435
    Early stopping, best iteration is:
    [3721]	valid_0's auc: 0.73845
    0.7384498877078078
    ************************************ 8 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.720998
    [400]	valid_0's auc: 0.726135
    [600]	valid_0's auc: 0.729196
    [800]	valid_0's auc: 0.731014
    [1000]	valid_0's auc: 0.732355
    [1200]	valid_0's auc: 0.733451
    [1400]	valid_0's auc: 0.734098
    [1600]	valid_0's auc: 0.734744
    [1800]	valid_0's auc: 0.735213
    [2000]	valid_0's auc: 0.735534
    [2200]	valid_0's auc: 0.735902
    [2400]	valid_0's auc: 0.736168
    [2600]	valid_0's auc: 0.736462
    [2800]	valid_0's auc: 0.736694
    [3000]	valid_0's auc: 0.73679
    [3200]	valid_0's auc: 0.737019
    [3400]	valid_0's auc: 0.737119
    [3600]	valid_0's auc: 0.737243
    [3800]	valid_0's auc: 0.737367
    [4000]	valid_0's auc: 0.737448
    [4200]	valid_0's auc: 0.73748
    [4400]	valid_0's auc: 0.737522
    [4600]	valid_0's auc: 0.737581
    [4800]	valid_0's auc: 0.737657
    [5000]	valid_0's auc: 0.737678
    [5200]	valid_0's auc: 0.737738
    [5400]	valid_0's auc: 0.737708
    Early stopping, best iteration is:
    [5335]	valid_0's auc: 0.737746
    0.7377458397660797
    ************************************ 9 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722888
    [400]	valid_0's auc: 0.727719
    [600]	valid_0's auc: 0.730287
    [800]	valid_0's auc: 0.732149
    [1000]	valid_0's auc: 0.733437
    [1200]	valid_0's auc: 0.734429
    [1400]	valid_0's auc: 0.735054
    [1600]	valid_0's auc: 0.735586
    [1800]	valid_0's auc: 0.735954
    [2000]	valid_0's auc: 0.736264
    [2200]	valid_0's auc: 0.736474
    [2400]	valid_0's auc: 0.736668
    [2600]	valid_0's auc: 0.736869
    [2800]	valid_0's auc: 0.737042
    [3000]	valid_0's auc: 0.737236
    [3200]	valid_0's auc: 0.737408
    [3400]	valid_0's auc: 0.737472
    [3600]	valid_0's auc: 0.737617
    [3800]	valid_0's auc: 0.737684
    [4000]	valid_0's auc: 0.73771
    [4200]	valid_0's auc: 0.737791
    [4400]	valid_0's auc: 0.737743
    Early stopping, best iteration is:
    [4207]	valid_0's auc: 0.737797
    0.7377973761699876
    ************************************ 10 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	valid_0's auc: 0.722192
    [400]	valid_0's auc: 0.726759
    [600]	valid_0's auc: 0.729255
    [800]	valid_0's auc: 0.731126
    [1000]	valid_0's auc: 0.732328
    [1200]	valid_0's auc: 0.733153
    [1400]	valid_0's auc: 0.733735
    [1600]	valid_0's auc: 0.734251
    [1800]	valid_0's auc: 0.734688
    [2000]	valid_0's auc: 0.734963
    [2200]	valid_0's auc: 0.735266
    [2400]	valid_0's auc: 0.735463
    [2600]	valid_0's auc: 0.735594
    [2800]	valid_0's auc: 0.735645
    [3000]	valid_0's auc: 0.735725
    [3200]	valid_0's auc: 0.73587
    [3400]	valid_0's auc: 0.735887
    [3600]	valid_0's auc: 0.735977
    [3800]	valid_0's auc: 0.735992
    [4000]	valid_0's auc: 0.736006
    [4200]	valid_0's auc: 0.736036
    [4400]	valid_0's auc: 0.736071
    [4600]	valid_0's auc: 0.736012
    Early stopping, best iteration is:
    [4452]	valid_0's auc: 0.736092
    0.7360919582991823
    


```python
result = pd.DataFrame({'id': data_test_a['id'], 'isDefault': test})
result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>isDefault</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>800000</td>
      <td>0.007390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>800001</td>
      <td>0.030056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>800002</td>
      <td>0.064912</td>
    </tr>
    <tr>
      <th>3</th>
      <td>800003</td>
      <td>0.031859</td>
    </tr>
    <tr>
      <th>4</th>
      <td>800004</td>
      <td>0.035386</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('result.csv', index=0)
```
