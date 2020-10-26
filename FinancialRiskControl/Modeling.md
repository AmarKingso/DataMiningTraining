导入所需库


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
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


```python
data_train = pd.read_csv('clean_train.csv')
data_train = reduce_mem_usage(data_train)
data_test_a = pd.read_csv('clean_test.csv')
data_test_a = reduce_mem_usage(data_test_a)
```

    Mem. usage decreased to 77.39 Mb (73.2% reduction)
    Mem. usage decreased to 19.46 Mb (72.9% reduction)
    


```python
features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault']]
train_x = data_train[features]
test_x = data_test_a[features]
train_y = data_train['isDefault']
```


```python
folds = 5
seed = 2020
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

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
        'min_child_weight': 5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.02,
        'seed': 2020,
        'nthread': 28,
        'n_jobs':24,
        'silent': True,
        'verbose': -1,
    }
    
    model = lgb.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200,early_stopping_rounds=200)
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_x, num_iteration=model.best_iteration)
    
    train[valid_index] = val_pred
    test = test_pred / kf.n_splits
    score = roc_auc_score(val_y, val_pred)

    print(score)
```

    ************************************ 1 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728929	valid_1's auc: 0.724579
    [400]	training's auc: 0.737417	valid_1's auc: 0.730075
    [600]	training's auc: 0.742355	valid_1's auc: 0.73233
    [800]	training's auc: 0.745938	valid_1's auc: 0.73332
    [1000]	training's auc: 0.749104	valid_1's auc: 0.73403
    [1200]	training's auc: 0.751956	valid_1's auc: 0.734458
    [1400]	training's auc: 0.754643	valid_1's auc: 0.734826
    [1600]	training's auc: 0.75731	valid_1's auc: 0.735201
    [1800]	training's auc: 0.75988	valid_1's auc: 0.735433
    [2000]	training's auc: 0.762261	valid_1's auc: 0.735604
    [2200]	training's auc: 0.764708	valid_1's auc: 0.735816
    [2400]	training's auc: 0.767064	valid_1's auc: 0.736015
    [2600]	training's auc: 0.769257	valid_1's auc: 0.736116
    [2800]	training's auc: 0.771381	valid_1's auc: 0.736142
    [3000]	training's auc: 0.773529	valid_1's auc: 0.73623
    [3200]	training's auc: 0.775665	valid_1's auc: 0.736255
    [3400]	training's auc: 0.777737	valid_1's auc: 0.736318
    [3600]	training's auc: 0.779743	valid_1's auc: 0.736296
    Early stopping, best iteration is:
    [3407]	training's auc: 0.777822	valid_1's auc: 0.736328
    0.736327867390652
    ************************************ 2 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.729221	valid_1's auc: 0.723222
    [400]	training's auc: 0.737475	valid_1's auc: 0.729144
    [600]	training's auc: 0.742471	valid_1's auc: 0.731598
    [800]	training's auc: 0.74612	valid_1's auc: 0.732743
    [1000]	training's auc: 0.749291	valid_1's auc: 0.73352
    [1200]	training's auc: 0.752228	valid_1's auc: 0.733957
    [1400]	training's auc: 0.754875	valid_1's auc: 0.734272
    [1600]	training's auc: 0.75746	valid_1's auc: 0.734454
    [1800]	training's auc: 0.759969	valid_1's auc: 0.734634
    [2000]	training's auc: 0.762461	valid_1's auc: 0.734849
    [2200]	training's auc: 0.764784	valid_1's auc: 0.734981
    [2400]	training's auc: 0.767063	valid_1's auc: 0.735165
    [2600]	training's auc: 0.769276	valid_1's auc: 0.73527
    [2800]	training's auc: 0.771502	valid_1's auc: 0.735368
    [3000]	training's auc: 0.773701	valid_1's auc: 0.735451
    [3200]	training's auc: 0.775775	valid_1's auc: 0.735462
    [3400]	training's auc: 0.777924	valid_1's auc: 0.735475
    [3600]	training's auc: 0.780016	valid_1's auc: 0.73554
    [3800]	training's auc: 0.781996	valid_1's auc: 0.735605
    [4000]	training's auc: 0.78405	valid_1's auc: 0.735707
    [4200]	training's auc: 0.785996	valid_1's auc: 0.735687
    Early stopping, best iteration is:
    [4028]	training's auc: 0.784339	valid_1's auc: 0.735721
    0.7357213799089078
    ************************************ 3 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728442	valid_1's auc: 0.726975
    [400]	training's auc: 0.736769	valid_1's auc: 0.732682
    [600]	training's auc: 0.741735	valid_1's auc: 0.735002
    [800]	training's auc: 0.745481	valid_1's auc: 0.735994
    [1000]	training's auc: 0.748662	valid_1's auc: 0.736593
    [1200]	training's auc: 0.751545	valid_1's auc: 0.737022
    [1400]	training's auc: 0.754246	valid_1's auc: 0.737357
    [1600]	training's auc: 0.756876	valid_1's auc: 0.737552
    [1800]	training's auc: 0.759399	valid_1's auc: 0.737761
    [2000]	training's auc: 0.761766	valid_1's auc: 0.737958
    [2200]	training's auc: 0.764238	valid_1's auc: 0.738162
    [2400]	training's auc: 0.766533	valid_1's auc: 0.738221
    [2600]	training's auc: 0.768762	valid_1's auc: 0.738269
    [2800]	training's auc: 0.770933	valid_1's auc: 0.738346
    [3000]	training's auc: 0.773048	valid_1's auc: 0.738342
    [3200]	training's auc: 0.775192	valid_1's auc: 0.738365
    Early stopping, best iteration is:
    [3108]	training's auc: 0.774229	valid_1's auc: 0.738391
    0.7383905887468065
    ************************************ 4 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728646	valid_1's auc: 0.725504
    [400]	training's auc: 0.736778	valid_1's auc: 0.73163
    [600]	training's auc: 0.741725	valid_1's auc: 0.734176
    [800]	training's auc: 0.745493	valid_1's auc: 0.735361
    [1000]	training's auc: 0.748715	valid_1's auc: 0.73609
    [1200]	training's auc: 0.751616	valid_1's auc: 0.736572
    [1400]	training's auc: 0.754381	valid_1's auc: 0.736974
    [1600]	training's auc: 0.757023	valid_1's auc: 0.737284
    [1800]	training's auc: 0.759547	valid_1's auc: 0.737461
    [2000]	training's auc: 0.76196	valid_1's auc: 0.737671
    [2200]	training's auc: 0.764325	valid_1's auc: 0.737766
    [2400]	training's auc: 0.766582	valid_1's auc: 0.737877
    [2600]	training's auc: 0.768774	valid_1's auc: 0.738011
    [2800]	training's auc: 0.770988	valid_1's auc: 0.738142
    [3000]	training's auc: 0.773168	valid_1's auc: 0.738183
    [3200]	training's auc: 0.775316	valid_1's auc: 0.738227
    [3400]	training's auc: 0.777425	valid_1's auc: 0.738303
    [3600]	training's auc: 0.779502	valid_1's auc: 0.738331
    [3800]	training's auc: 0.781498	valid_1's auc: 0.738337
    Early stopping, best iteration is:
    [3660]	training's auc: 0.780128	valid_1's auc: 0.738373
    0.7383727610331814
    ************************************ 5 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728617	valid_1's auc: 0.726112
    [400]	training's auc: 0.736843	valid_1's auc: 0.73144
    [600]	training's auc: 0.741854	valid_1's auc: 0.733669
    [800]	training's auc: 0.745588	valid_1's auc: 0.734727
    [1000]	training's auc: 0.748782	valid_1's auc: 0.735385
    [1200]	training's auc: 0.751651	valid_1's auc: 0.735786
    [1400]	training's auc: 0.754494	valid_1's auc: 0.736192
    [1600]	training's auc: 0.757063	valid_1's auc: 0.736463
    [1800]	training's auc: 0.759624	valid_1's auc: 0.736715
    [2000]	training's auc: 0.762046	valid_1's auc: 0.736923
    [2200]	training's auc: 0.764448	valid_1's auc: 0.737008
    [2400]	training's auc: 0.766714	valid_1's auc: 0.73716
    [2600]	training's auc: 0.768979	valid_1's auc: 0.737218
    [2800]	training's auc: 0.771207	valid_1's auc: 0.737343
    [3000]	training's auc: 0.773381	valid_1's auc: 0.73736
    [3200]	training's auc: 0.775542	valid_1's auc: 0.737424
    [3400]	training's auc: 0.777653	valid_1's auc: 0.737466
    [3600]	training's auc: 0.779709	valid_1's auc: 0.737567
    [3800]	training's auc: 0.781818	valid_1's auc: 0.737607
    [4000]	training's auc: 0.783833	valid_1's auc: 0.737636
    Early stopping, best iteration is:
    [3932]	training's auc: 0.783161	valid_1's auc: 0.737646
    0.7376457539556787
    


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
      <td>0.013911</td>
    </tr>
    <tr>
      <th>1</th>
      <td>800001</td>
      <td>0.058692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>800002</td>
      <td>0.127429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>800003</td>
      <td>0.064085</td>
    </tr>
    <tr>
      <th>4</th>
      <td>800004</td>
      <td>0.080466</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('result.csv', index=0)
```
