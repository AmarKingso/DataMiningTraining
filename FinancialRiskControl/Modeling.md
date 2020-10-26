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

采用lightgbm建模，k折交叉验证来评估模型


```python
folds = 10
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
    [200]	training's auc: 0.728274	valid_1's auc: 0.728114
    [400]	training's auc: 0.736254	valid_1's auc: 0.733037
    [600]	training's auc: 0.741072	valid_1's auc: 0.735455
    [800]	training's auc: 0.744507	valid_1's auc: 0.736515
    [1000]	training's auc: 0.747471	valid_1's auc: 0.737121
    [1200]	training's auc: 0.75007	valid_1's auc: 0.737517
    [1400]	training's auc: 0.752585	valid_1's auc: 0.737877
    [1600]	training's auc: 0.755072	valid_1's auc: 0.738193
    [1800]	training's auc: 0.757413	valid_1's auc: 0.738361
    [2000]	training's auc: 0.759641	valid_1's auc: 0.73847
    [2200]	training's auc: 0.761764	valid_1's auc: 0.7386
    [2400]	training's auc: 0.763913	valid_1's auc: 0.738681
    [2600]	training's auc: 0.766015	valid_1's auc: 0.738804
    [2800]	training's auc: 0.768041	valid_1's auc: 0.738984
    [3000]	training's auc: 0.770022	valid_1's auc: 0.739017
    [3200]	training's auc: 0.771961	valid_1's auc: 0.739047
    [3400]	training's auc: 0.773833	valid_1's auc: 0.739157
    [3600]	training's auc: 0.775644	valid_1's auc: 0.739228
    [3800]	training's auc: 0.777574	valid_1's auc: 0.739259
    [4000]	training's auc: 0.779426	valid_1's auc: 0.739311
    [4200]	training's auc: 0.781228	valid_1's auc: 0.739314
    Early stopping, best iteration is:
    [4118]	training's auc: 0.780509	valid_1's auc: 0.739361
    0.7393609990534565
    ************************************ 2 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728899	valid_1's auc: 0.721203
    [400]	training's auc: 0.736961	valid_1's auc: 0.726921
    [600]	training's auc: 0.74165	valid_1's auc: 0.729371
    [800]	training's auc: 0.745093	valid_1's auc: 0.730578
    [1000]	training's auc: 0.748068	valid_1's auc: 0.731353
    [1200]	training's auc: 0.750693	valid_1's auc: 0.731783
    [1400]	training's auc: 0.753242	valid_1's auc: 0.732196
    [1600]	training's auc: 0.755662	valid_1's auc: 0.732593
    [1800]	training's auc: 0.757999	valid_1's auc: 0.732744
    [2000]	training's auc: 0.760187	valid_1's auc: 0.732986
    [2200]	training's auc: 0.762233	valid_1's auc: 0.733179
    [2400]	training's auc: 0.764317	valid_1's auc: 0.733392
    [2600]	training's auc: 0.766346	valid_1's auc: 0.733575
    [2800]	training's auc: 0.768413	valid_1's auc: 0.733675
    [3000]	training's auc: 0.770301	valid_1's auc: 0.733793
    [3200]	training's auc: 0.772269	valid_1's auc: 0.733891
    [3400]	training's auc: 0.774157	valid_1's auc: 0.733946
    [3600]	training's auc: 0.775984	valid_1's auc: 0.734005
    [3800]	training's auc: 0.777796	valid_1's auc: 0.73406
    [4000]	training's auc: 0.779644	valid_1's auc: 0.734131
    [4200]	training's auc: 0.781422	valid_1's auc: 0.734176
    [4400]	training's auc: 0.783219	valid_1's auc: 0.734176
    [4600]	training's auc: 0.785078	valid_1's auc: 0.734242
    [4800]	training's auc: 0.786854	valid_1's auc: 0.73431
    [5000]	training's auc: 0.788527	valid_1's auc: 0.734394
    Early stopping, best iteration is:
    [4995]	training's auc: 0.788482	valid_1's auc: 0.734404
    0.7344039385597075
    ************************************ 3 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728669	valid_1's auc: 0.722433
    [400]	training's auc: 0.736707	valid_1's auc: 0.728799
    [600]	training's auc: 0.741467	valid_1's auc: 0.731669
    [800]	training's auc: 0.744887	valid_1's auc: 0.732987
    [1000]	training's auc: 0.747853	valid_1's auc: 0.733827
    [1200]	training's auc: 0.750441	valid_1's auc: 0.734305
    [1400]	training's auc: 0.753023	valid_1's auc: 0.734726
    [1600]	training's auc: 0.755304	valid_1's auc: 0.734977
    [1800]	training's auc: 0.757642	valid_1's auc: 0.735288
    [2000]	training's auc: 0.759853	valid_1's auc: 0.735557
    [2200]	training's auc: 0.76195	valid_1's auc: 0.735707
    [2400]	training's auc: 0.764071	valid_1's auc: 0.735911
    [2600]	training's auc: 0.766109	valid_1's auc: 0.736143
    [2800]	training's auc: 0.768098	valid_1's auc: 0.736211
    [3000]	training's auc: 0.770043	valid_1's auc: 0.73624
    [3200]	training's auc: 0.772042	valid_1's auc: 0.736319
    [3400]	training's auc: 0.774058	valid_1's auc: 0.736404
    [3600]	training's auc: 0.775953	valid_1's auc: 0.7365
    [3800]	training's auc: 0.777817	valid_1's auc: 0.736611
    [4000]	training's auc: 0.779632	valid_1's auc: 0.736665
    [4200]	training's auc: 0.781412	valid_1's auc: 0.73675
    [4400]	training's auc: 0.78325	valid_1's auc: 0.736767
    [4600]	training's auc: 0.785086	valid_1's auc: 0.736867
    [4800]	training's auc: 0.786789	valid_1's auc: 0.736861
    Early stopping, best iteration is:
    [4617]	training's auc: 0.785218	valid_1's auc: 0.736875
    0.7368753150559509
    ************************************ 4 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728636	valid_1's auc: 0.723899
    [400]	training's auc: 0.736796	valid_1's auc: 0.729332
    [600]	training's auc: 0.741563	valid_1's auc: 0.731656
    [800]	training's auc: 0.745035	valid_1's auc: 0.732734
    [1000]	training's auc: 0.748005	valid_1's auc: 0.733502
    [1200]	training's auc: 0.750607	valid_1's auc: 0.733878
    [1400]	training's auc: 0.753095	valid_1's auc: 0.734199
    [1600]	training's auc: 0.755479	valid_1's auc: 0.734489
    [1800]	training's auc: 0.757716	valid_1's auc: 0.734655
    [2000]	training's auc: 0.759946	valid_1's auc: 0.73482
    [2200]	training's auc: 0.762167	valid_1's auc: 0.735
    [2400]	training's auc: 0.764244	valid_1's auc: 0.735177
    [2600]	training's auc: 0.766245	valid_1's auc: 0.735302
    [2800]	training's auc: 0.768192	valid_1's auc: 0.735376
    [3000]	training's auc: 0.77019	valid_1's auc: 0.735511
    [3200]	training's auc: 0.772159	valid_1's auc: 0.735595
    [3400]	training's auc: 0.773993	valid_1's auc: 0.735657
    [3600]	training's auc: 0.775878	valid_1's auc: 0.735752
    [3800]	training's auc: 0.777702	valid_1's auc: 0.735785
    [4000]	training's auc: 0.779585	valid_1's auc: 0.735864
    [4200]	training's auc: 0.781405	valid_1's auc: 0.73593
    [4400]	training's auc: 0.783204	valid_1's auc: 0.735904
    Early stopping, best iteration is:
    [4244]	training's auc: 0.781804	valid_1's auc: 0.735949
    0.7359490459219475
    ************************************ 5 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728283	valid_1's auc: 0.725978
    [400]	training's auc: 0.736489	valid_1's auc: 0.731636
    [600]	training's auc: 0.741319	valid_1's auc: 0.733941
    [800]	training's auc: 0.744769	valid_1's auc: 0.734996
    [1000]	training's auc: 0.747672	valid_1's auc: 0.735633
    [1200]	training's auc: 0.750361	valid_1's auc: 0.736096
    [1400]	training's auc: 0.752816	valid_1's auc: 0.736358
    [1600]	training's auc: 0.755219	valid_1's auc: 0.736614
    [1800]	training's auc: 0.75754	valid_1's auc: 0.736735
    [2000]	training's auc: 0.759728	valid_1's auc: 0.73694
    [2200]	training's auc: 0.76193	valid_1's auc: 0.737062
    [2400]	training's auc: 0.764009	valid_1's auc: 0.737165
    [2600]	training's auc: 0.76608	valid_1's auc: 0.73718
    [2800]	training's auc: 0.768174	valid_1's auc: 0.737259
    [3000]	training's auc: 0.770202	valid_1's auc: 0.737315
    [3200]	training's auc: 0.772146	valid_1's auc: 0.737341
    [3400]	training's auc: 0.774048	valid_1's auc: 0.73738
    [3600]	training's auc: 0.775862	valid_1's auc: 0.737456
    Early stopping, best iteration is:
    [3553]	training's auc: 0.775464	valid_1's auc: 0.737488
    0.7374883367093499
    ************************************ 6 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728146	valid_1's auc: 0.727839
    [400]	training's auc: 0.736173	valid_1's auc: 0.733644
    [600]	training's auc: 0.740906	valid_1's auc: 0.736147
    [800]	training's auc: 0.74432	valid_1's auc: 0.737225
    [1000]	training's auc: 0.747264	valid_1's auc: 0.737852
    [1200]	training's auc: 0.749933	valid_1's auc: 0.738238
    [1400]	training's auc: 0.752425	valid_1's auc: 0.738567
    [1600]	training's auc: 0.754837	valid_1's auc: 0.738809
    [1800]	training's auc: 0.757157	valid_1's auc: 0.739009
    [2000]	training's auc: 0.759369	valid_1's auc: 0.73926
    [2200]	training's auc: 0.761593	valid_1's auc: 0.739405
    [2400]	training's auc: 0.763761	valid_1's auc: 0.739462
    [2600]	training's auc: 0.765747	valid_1's auc: 0.739498
    [2800]	training's auc: 0.76773	valid_1's auc: 0.739617
    [3000]	training's auc: 0.769669	valid_1's auc: 0.739725
    [3200]	training's auc: 0.771576	valid_1's auc: 0.739741
    [3400]	training's auc: 0.77348	valid_1's auc: 0.739805
    [3600]	training's auc: 0.775456	valid_1's auc: 0.739914
    Early stopping, best iteration is:
    [3592]	training's auc: 0.775373	valid_1's auc: 0.739931
    0.7399307528251254
    ************************************ 7 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728177	valid_1's auc: 0.72686
    [400]	training's auc: 0.736229	valid_1's auc: 0.733394
    [600]	training's auc: 0.740991	valid_1's auc: 0.736037
    [800]	training's auc: 0.744468	valid_1's auc: 0.73729
    [1000]	training's auc: 0.747416	valid_1's auc: 0.737899
    [1200]	training's auc: 0.75003	valid_1's auc: 0.738318
    [1400]	training's auc: 0.752441	valid_1's auc: 0.738638
    [1600]	training's auc: 0.754837	valid_1's auc: 0.738844
    [1800]	training's auc: 0.75719	valid_1's auc: 0.739139
    [2000]	training's auc: 0.759334	valid_1's auc: 0.739261
    [2200]	training's auc: 0.761502	valid_1's auc: 0.739478
    [2400]	training's auc: 0.763569	valid_1's auc: 0.739588
    [2600]	training's auc: 0.765686	valid_1's auc: 0.739744
    [2800]	training's auc: 0.767801	valid_1's auc: 0.739845
    [3000]	training's auc: 0.769852	valid_1's auc: 0.739915
    [3200]	training's auc: 0.771746	valid_1's auc: 0.740063
    [3400]	training's auc: 0.773655	valid_1's auc: 0.740177
    [3600]	training's auc: 0.775544	valid_1's auc: 0.74015
    Early stopping, best iteration is:
    [3534]	training's auc: 0.774915	valid_1's auc: 0.740205
    0.7402049771405007
    ************************************ 8 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728549	valid_1's auc: 0.724425
    [400]	training's auc: 0.736582	valid_1's auc: 0.730524
    [600]	training's auc: 0.741321	valid_1's auc: 0.733028
    [800]	training's auc: 0.744738	valid_1's auc: 0.734236
    [1000]	training's auc: 0.747637	valid_1's auc: 0.734918
    [1200]	training's auc: 0.750329	valid_1's auc: 0.73531
    [1400]	training's auc: 0.752885	valid_1's auc: 0.735753
    [1600]	training's auc: 0.755181	valid_1's auc: 0.736088
    [1800]	training's auc: 0.757374	valid_1's auc: 0.736318
    [2000]	training's auc: 0.759637	valid_1's auc: 0.736563
    [2200]	training's auc: 0.761773	valid_1's auc: 0.736709
    [2400]	training's auc: 0.76388	valid_1's auc: 0.73691
    [2600]	training's auc: 0.765935	valid_1's auc: 0.736918
    [2800]	training's auc: 0.768035	valid_1's auc: 0.737022
    [3000]	training's auc: 0.770036	valid_1's auc: 0.737058
    [3200]	training's auc: 0.771959	valid_1's auc: 0.737154
    [3400]	training's auc: 0.773937	valid_1's auc: 0.737186
    [3600]	training's auc: 0.775875	valid_1's auc: 0.737316
    [3800]	training's auc: 0.777692	valid_1's auc: 0.737341
    [4000]	training's auc: 0.779533	valid_1's auc: 0.737351
    Early stopping, best iteration is:
    [3889]	training's auc: 0.778582	valid_1's auc: 0.737396
    0.737396349439526
    ************************************ 9 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728216	valid_1's auc: 0.726974
    [400]	training's auc: 0.736403	valid_1's auc: 0.7325
    [600]	training's auc: 0.741124	valid_1's auc: 0.734718
    [800]	training's auc: 0.744634	valid_1's auc: 0.735851
    [1000]	training's auc: 0.747547	valid_1's auc: 0.736438
    [1200]	training's auc: 0.750188	valid_1's auc: 0.736769
    [1400]	training's auc: 0.752755	valid_1's auc: 0.737113
    [1600]	training's auc: 0.75512	valid_1's auc: 0.737315
    [1800]	training's auc: 0.757336	valid_1's auc: 0.737483
    [2000]	training's auc: 0.759549	valid_1's auc: 0.737659
    [2200]	training's auc: 0.761787	valid_1's auc: 0.737808
    [2400]	training's auc: 0.763916	valid_1's auc: 0.737924
    [2600]	training's auc: 0.766003	valid_1's auc: 0.73806
    [2800]	training's auc: 0.76805	valid_1's auc: 0.738161
    [3000]	training's auc: 0.77004	valid_1's auc: 0.738203
    [3200]	training's auc: 0.77194	valid_1's auc: 0.738251
    [3400]	training's auc: 0.773899	valid_1's auc: 0.738268
    [3600]	training's auc: 0.77576	valid_1's auc: 0.738272
    Early stopping, best iteration is:
    [3476]	training's auc: 0.774627	valid_1's auc: 0.738314
    0.7383138756774676
    ************************************ 10 ************************************
    Training until validation scores don't improve for 200 rounds
    [200]	training's auc: 0.728483	valid_1's auc: 0.725435
    [400]	training's auc: 0.736491	valid_1's auc: 0.73065
    [600]	training's auc: 0.741313	valid_1's auc: 0.732941
    [800]	training's auc: 0.744796	valid_1's auc: 0.734065
    [1000]	training's auc: 0.747765	valid_1's auc: 0.734712
    [1200]	training's auc: 0.750416	valid_1's auc: 0.735232
    [1400]	training's auc: 0.752971	valid_1's auc: 0.735688
    [1600]	training's auc: 0.75532	valid_1's auc: 0.735875
    [1800]	training's auc: 0.757652	valid_1's auc: 0.7361
    [2000]	training's auc: 0.759852	valid_1's auc: 0.736231
    [2200]	training's auc: 0.76209	valid_1's auc: 0.736448
    [2400]	training's auc: 0.764093	valid_1's auc: 0.736609
    [2600]	training's auc: 0.766098	valid_1's auc: 0.736688
    [2800]	training's auc: 0.768094	valid_1's auc: 0.736741
    [3000]	training's auc: 0.770073	valid_1's auc: 0.736856
    [3200]	training's auc: 0.772057	valid_1's auc: 0.73694
    [3400]	training's auc: 0.77398	valid_1's auc: 0.737043
    [3600]	training's auc: 0.775851	valid_1's auc: 0.73712
    [3800]	training's auc: 0.7778	valid_1's auc: 0.737118
    Early stopping, best iteration is:
    [3667]	training's auc: 0.77651	valid_1's auc: 0.737133
    0.7371329574117977
    


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
      <td>0.007129</td>
    </tr>
    <tr>
      <th>1</th>
      <td>800001</td>
      <td>0.032222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>800002</td>
      <td>0.060530</td>
    </tr>
    <tr>
      <th>3</th>
      <td>800003</td>
      <td>0.030170</td>
    </tr>
    <tr>
      <th>4</th>
      <td>800004</td>
      <td>0.038801</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('result.csv', index=0)
```
