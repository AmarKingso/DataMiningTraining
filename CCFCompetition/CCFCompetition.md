# 企业非法集资风险预测

导入所需的库


```python
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import f1_score,precision_recall_fscore_support
from bayes_opt import BayesianOptimization
import warnings
import jieba
from zhon.hanzi import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
```

## 数据分析

读取所有文件，导入数据集


```python
base_info = pd.read_csv('base_info.csv')
annual_report_info = pd.read_csv('annual_report_info.csv')
tax_info = pd.read_csv('tax_info.csv')
change_info = pd.read_csv('change_info.csv')
news_info = pd.read_csv('news_info.csv')
other_info = pd.read_csv('other_info.csv')
entprise_info = pd.read_csv('entprise_info.csv')
entprise_evaluate = pd.read_csv('entprise_evaluate.csv')
```

根据提供的信息，可知最后两个分别为训练集标签与待预测的验证集标签，只有id与标签两列

各数据集样本数量以及所包含的企业数量


```python
print('base_info shape: {}; base_info unique: {}'.format(base_info.shape, len(base_info['id'].unique())))
print('annual_report_info shape: {}; annual_report_info unique: {}'.format(annual_report_info.shape, len(annual_report_info['id'].unique())))
print('tax_info shape: {}; tax_info unique: {}'.format(tax_info.shape, len(tax_info['id'].unique())))
print('change_info shape: {}; change_info unique: {}'.format(change_info.shape, len(change_info['id'].unique())))
print('news_info shape: {}; news_info unique: {}'.format(news_info.shape, len(news_info['id'].unique())))
print('other_info shape: {}; other_info unique: {}'.format(other_info.shape, len(other_info['id'].unique())))
print('entprise_info shape: {}; entprise_info unique: {}'.format(entprise_info.shape, len(entprise_info['id'].unique())))
print('entprise_evaluate shape: {}; entprise_evaluate unique: {}'.format(entprise_evaluate.shape, len(entprise_evaluate['id'].unique())))
```

    base_info shape: (24865, 33); base_info unique: 24865
    annual_report_info shape: (22550, 23); annual_report_info unique: 8937
    tax_info shape: (29195, 9); tax_info unique: 808
    change_info shape: (45940, 5); change_info unique: 8726
    news_info shape: (10518, 3); news_info unique: 927
    other_info shape: (1890, 4); other_info unique: 1888
    entprise_info shape: (14865, 2); entprise_info unique: 14865
    entprise_evaluate shape: (10000, 2); entprise_evaluate unique: 10000
    

从以上信息可以看出，整个数据集共有24865家企业，除了base_info完整地提供了各企业的基本信息，其他数据集的企业附加信息并不完整，有很多企业并没有提供，类似新闻信息这一类更是缺失严重。除此之外，我们还能看到作为测试集的样本有14865条，作为验证集的有10000条

查看base_info数据集的信息


```python
base_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24865 entries, 0 to 24864
    Data columns (total 33 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             24865 non-null  object 
     1   oplocdistrict  24865 non-null  int64  
     2   industryphy    24865 non-null  object 
     3   industryco     24864 non-null  float64
     4   dom            24865 non-null  object 
     5   opscope        24865 non-null  object 
     6   enttype        24865 non-null  int64  
     7   enttypeitem    16651 non-null  float64
     8   opfrom         24865 non-null  object 
     9   opto           8825 non-null   object 
     10  state          24865 non-null  int64  
     11  orgid          24865 non-null  int64  
     12  jobid          24865 non-null  int64  
     13  adbusign       24865 non-null  int64  
     14  townsign       24865 non-null  int64  
     15  regtype        24865 non-null  int64  
     16  empnum         19615 non-null  float64
     17  compform       10631 non-null  float64
     18  parnum         2339 non-null   float64
     19  exenum         1378 non-null   float64
     20  opform         9000 non-null   object 
     21  ptbusscope     0 non-null      float64
     22  venind         8437 non-null   float64
     23  enttypeminu    7270 non-null   float64
     24  midpreindcode  0 non-null      float64
     25  protype        34 non-null     float64
     26  oploc          24865 non-null  object 
     27  regcap         24674 non-null  float64
     28  reccap         7084 non-null   float64
     29  forreccap      227 non-null    float64
     30  forregcap      250 non-null    float64
     31  congro         249 non-null    float64
     32  enttypegb      24865 non-null  int64  
    dtypes: float64(16), int64(9), object(8)
    memory usage: 6.3+ MB
    

如上述所言，除了企业基本信息外，其余数据集企业空缺都比较大，所以最先考虑只用base_info一个数据集来建模。

## 数据预处理

### base_info

查看base_info各特征缺失率，并将缺失率超过0.5的特征删除


```python
base_clean = base_info.dropna(thresh=base_info.shape[0]*0.5, how='all', axis=1)
```

查看拥有唯一值的特征


```python
unique_val = [fea for fea in base_clean.columns if base_clean[fea].nunique() <= 1]
unique_val
```




    []



特征数据类型划分


```python
num_fea = list(base_clean.select_dtypes(exclude=['object']).columns)
num_fea
```




    ['oplocdistrict',
     'industryco',
     'enttype',
     'enttypeitem',
     'state',
     'orgid',
     'jobid',
     'adbusign',
     'townsign',
     'regtype',
     'empnum',
     'regcap',
     'enttypegb']




```python
obj_fea = list(filter(lambda x: x not in num_fea,list(base_clean.columns)))
obj_fea
```




    ['id', 'industryphy', 'dom', 'opscope', 'opfrom', 'oploc']



这一步我们需要做的是将对象型特征转化为数值型，以便后续缺失值的填补以及其他操作。

首先我们要了解这几个对象型特征的含义，id不用管，该特征对最后建模没有影响；industryphy为行业类别代码，dom为经营地址；opscope为经营范围；opfrom为经营期限起，oploc为经营场所

查看这几个属性的具体特征值


```python
base_clean[obj_fea].head()
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
      <th>industryphy</th>
      <th>dom</th>
      <th>opscope</th>
      <th>opfrom</th>
      <th>oploc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47645761dc56bb8c5fae00114b768b5d9b6e917c3aec07c4</td>
      <td>M</td>
      <td>31487d8f256f16bd6244b7251be2ebb24d1db51663c654...</td>
      <td>纳米新材料、机械设备、五金配件加工、销售及技术推广服务，道路货物运输。（依法须经批准的项目，...</td>
      <td>2019-07-11 00:00:00</td>
      <td>2367b4cac96d8598</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>O</td>
      <td>31487d8f256f16bd6244b7251be2ebb27b17bdfd95c8f3...</td>
      <td>健身服务。（依法须经批准的项目，经相关部门批准后方可开展经营活动）</td>
      <td>2017-09-06</td>
      <td>31487d8f256f16bd6244b7251be2ebb27b17bdfd95c8f3...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59b38c56de3836838082cfcb1a298951abfe15e6940c49ba</td>
      <td>R</td>
      <td>31487d8f256f16bd6244b7251be2ebb2ae36cd652943e8...</td>
      <td>文化娱乐经纪人服务；境内文艺活动组织与策划；文化艺术交流活动组织策划；演出经纪；其他文化艺术...</td>
      <td>2020-09-14 14:46:30</td>
      <td>2367b4cac96d8598</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e9f7b28ec10e047000d16ab79e1b5e6da434a1697cce7818</td>
      <td>L</td>
      <td>746df9aaed8578571760c563abe882c8ba25209fc6d5db...</td>
      <td>投资管理及咨询(证券、期货除外)；企业管理。（依法须经批准的项目，经相关部门批准后方可开展经...</td>
      <td>2015-09-30</td>
      <td>2367b4cac96d8598</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f000950527a6feb63ee1ce82bb22ddd1ab8b8fdffa3b91fb</td>
      <td>R</td>
      <td>31487d8f256f16bd6244b7251be2ebb2ae36cd652943e8...</td>
      <td>境内文化艺术交流活动策划；企业形象策划；礼仪庆典服务；翻译服务；专利代理；广告设计、制作、代...</td>
      <td>2017-12-01</td>
      <td>2367b4cac96d8598</td>
    </tr>
  </tbody>
</table>
</div>



- 对industryphy进行处理


```python
base_clean['industryphy'].unique()
```




    array(['M', 'O', 'R', 'L', 'P', 'J', 'Q', 'N', 'F', 'E', 'C', 'K', 'D',
           'I', 'S', 'G', 'A', 'T', 'H', 'B'], dtype=object)



可知该特征值是离散的，且相互之间不存在大小关系，采用one-hot映射


```python
base_clean = base_clean.join(pd.get_dummies(base_clean['industryphy'], prefix='industryphy'))
base_clean.drop('industryphy', axis=1, inplace=True)
```

- 对dom进行处理


```python
len(base_clean['dom'].unique())
```




    23278



该特征暂时没有好的转换方式，故采用直接映射的方法


```python
index = list(set(base_clean['dom']))
base_clean['dom'] = base_clean['dom'].map(dict(zip(index,list(range(len(index))))))
```

- 对opscope处理

该特征由汉字构成，考虑使用文本处理的那一套方法，即用tf-idf

使用jieba中文词库对文本进行分词，并去掉标点符号，然后对处理好的词集进行训练


```python
all_opscope = base_clean['opscope']
seg_words = [list(jieba.cut(sent)) for sent in all_opscope]
document = [" ".join(sent) for sent in seg_words]
# 去除标点符号
for i in range(len(document)):
    for j in punctuation:
        document[i] = document[i].replace(j,'')

# tf-idf统计词频
vectorizer = TfidfVectorizer().fit_transform(document).todense()
```

经过上面的过程，就将每个样本的opscope属性向量化了，然后我们要做的是对数据进行降维，这里我采用的是PCA + T-SNE的降维方法

PCA降维，剩余维度方差在95%以上


```python
pca = PCA(n_components=0.95, random_state=2020)
vect_reduced = pca.fit_transform(vectorizer)
```


```python
vect_reduced.shape
```




    (24865, 2186)



T-SNE将向量降至二维


```python
tsne = TSNE(random_state=2020)
weight = tsne.fit_transform(vect_reduced)
```

最后得到的二维向量列表


```python
weight
```




    array([[ -2.780353 ,  -8.698352 ],
           [ 36.049282 ,  15.989389 ],
           [ 11.963151 ,  50.096863 ],
           ...,
           [ -4.8659506, -60.46656  ],
           [-23.718998 ,   1.000325 ],
           [  2.7640297,  27.507668 ]], dtype=float32)



然后就是如何用二维表达出一维了，我图方便就直接相除了，也可以尝试平方和之类的对比效果


```python
base_clean['opscope']=weight[:,0]/weight[:,1]
base_clean['opscope']
```




    0         0.319641
    1         2.254575
    2         0.238800
    3        -1.025108
    4         0.076024
               ...    
    24860     0.698635
    24861    -1.480714
    24862     0.080473
    24863   -23.711292
    24864     0.100482
    Name: opscope, Length: 24865, dtype: float32



- 对opfrom进行处理

可以将该特征转化为与某一特定时间之间的时间差。同时观察数据可以得知，大多数企业的经营区间都是50年，由此可以大致推算填补opto的缺失值，不过这里先不实现，等提交过后再来进行该处理看分数是否有提高。


```python
base_clean['opfrom'] = pd.to_datetime(base_clean['opfrom'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2020-10-13', '%Y-%m-%d')
base_clean['opfromDays'] = base_clean['opfrom'].apply(lambda x: startdate-x).dt.days
# 填补opto
base_clean['opto'] = base_info['opto']
base_clean['opto'] = base_clean['opto'].fillna(pd.to_datetime(base_clean['opto']).max())
base_clean['opto'] = pd.to_datetime(base_clean['opto'],format='%Y-%m-%d')
base_clean['BaseGapDay'] = (base_clean['opto'] - base_clean['opfrom']).dt.days
base_clean.drop(['opfrom','opto'], axis=1, inplace=True)
```

- 对oploc进行处理

做和dom一样的处理，直接映射


```python
index = list(set(base_clean['oploc']))
base_clean['oploc'] = base_clean['oploc'].map(dict(zip(index,list(range(len(index))))))
```

至此特征类型的简单编码就完成了，查看处理后的数据


```python
base_clean.head()
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
      <th>oplocdistrict</th>
      <th>industryco</th>
      <th>dom</th>
      <th>opscope</th>
      <th>enttype</th>
      <th>enttypeitem</th>
      <th>state</th>
      <th>orgid</th>
      <th>jobid</th>
      <th>adbusign</th>
      <th>townsign</th>
      <th>regtype</th>
      <th>empnum</th>
      <th>oploc</th>
      <th>regcap</th>
      <th>enttypegb</th>
      <th>industryphy_A</th>
      <th>industryphy_B</th>
      <th>industryphy_C</th>
      <th>industryphy_D</th>
      <th>industryphy_E</th>
      <th>industryphy_F</th>
      <th>industryphy_G</th>
      <th>industryphy_H</th>
      <th>industryphy_I</th>
      <th>industryphy_J</th>
      <th>industryphy_K</th>
      <th>industryphy_L</th>
      <th>industryphy_M</th>
      <th>industryphy_N</th>
      <th>industryphy_O</th>
      <th>industryphy_P</th>
      <th>industryphy_Q</th>
      <th>industryphy_R</th>
      <th>industryphy_S</th>
      <th>industryphy_T</th>
      <th>opfromDays</th>
      <th>BaseGapDay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47645761dc56bb8c5fae00114b768b5d9b6e917c3aec07c4</td>
      <td>340223</td>
      <td>7513.0</td>
      <td>2999</td>
      <td>0.319641</td>
      <td>1100</td>
      <td>1150.0</td>
      <td>6</td>
      <td>340223010010000000</td>
      <td>340200000000115392</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5.0</td>
      <td>1062</td>
      <td>50.0</td>
      <td>1151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>460</td>
      <td>36011</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>340222</td>
      <td>8090.0</td>
      <td>3942</td>
      <td>2.254575</td>
      <td>9600</td>
      <td>NaN</td>
      <td>6</td>
      <td>340222060010000000</td>
      <td>340200000000112114</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>438</td>
      <td>10.0</td>
      <td>9600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1133</td>
      <td>36684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59b38c56de3836838082cfcb1a298951abfe15e6940c49ba</td>
      <td>340202</td>
      <td>9053.0</td>
      <td>4413</td>
      <td>0.238800</td>
      <td>1100</td>
      <td>1150.0</td>
      <td>6</td>
      <td>340202010010000000</td>
      <td>400000000000753910</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>1062</td>
      <td>100.0</td>
      <td>1151</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>35579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e9f7b28ec10e047000d16ab79e1b5e6da434a1697cce7818</td>
      <td>340221</td>
      <td>7212.0</td>
      <td>22523</td>
      <td>-1.025108</td>
      <td>4500</td>
      <td>4540.0</td>
      <td>6</td>
      <td>340221010010000000</td>
      <td>400000000000013538</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>1062</td>
      <td>10.0</td>
      <td>4540</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1840</td>
      <td>37391</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f000950527a6feb63ee1ce82bb22ddd1ab8b8fdffa3b91fb</td>
      <td>340202</td>
      <td>8810.0</td>
      <td>22847</td>
      <td>0.076024</td>
      <td>1100</td>
      <td>1130.0</td>
      <td>7</td>
      <td>340200000000000000</td>
      <td>400000000000283237</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>1062</td>
      <td>100.0</td>
      <td>1130</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1047</td>
      <td>18261</td>
    </tr>
  </tbody>
</table>
</div>



经观察，enttype，enttypeitem，enttypeminu和enttypegb，四个特征都是对企业类型的描述，前三个从左到右细分程度依次增高，而enttypegb包含了前三者的全部数据，所以前三个特征可以去除


```python
base_clean.drop(['enttype','enttypeitem'], axis=1, inplace=True)
```

添加回重要特征reccap


```python
base_clean['reccap'] = base_info['reccap']
base_clean['regcap_reccap'] = base_info['regcap'] - base_info['reccap']
```

用平均数填补缺失值


```python
base_clean = base_clean.fillna(base_clean.median())
```

## 融合数据

上一周只使用了一个base_info数据集建模，这次将其他几个数据集都融合到data_set中

### annual_report_info

查看annual_report_info信息


```python
annual_report_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 22550 entries, 0 to 22549
    Data columns (total 23 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             22550 non-null  object 
     1   ANCHEYEAR      22550 non-null  float64
     2   STATE          22545 non-null  float64
     3   FUNDAM         5702 non-null   float64
     4   MEMNUM         29 non-null     float64
     5   FARNUM         29 non-null     float64
     6   ANNNEWMEMNUM   29 non-null     float64
     7   ANNREDMEMNUM   29 non-null     float64
     8   EMPNUM         22535 non-null  float64
     9   EMPNUMSIGN     16833 non-null  float64
     10  BUSSTNAME      17680 non-null  object 
     11  COLGRANUM      20041 non-null  float64
     12  RETSOLNUM      20041 non-null  float64
     13  DISPERNUM      20041 non-null  float64
     14  UNENUM         20041 non-null  float64
     15  COLEMPLNUM     20041 non-null  float64
     16  RETEMPLNUM     20041 non-null  float64
     17  DISEMPLNUM     20041 non-null  float64
     18  UNEEMPLNUM     20041 non-null  float64
     19  WEBSITSIGN     22517 non-null  float64
     20  FORINVESTSIGN  16489 non-null  float64
     21  STOCKTRANSIGN  13507 non-null  float64
     22  PUBSTATE       22530 non-null  float64
    dtypes: float64(21), object(2)
    memory usage: 4.0+ MB
    


```python
annual_report_info.head()
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
      <th>ANCHEYEAR</th>
      <th>STATE</th>
      <th>FUNDAM</th>
      <th>MEMNUM</th>
      <th>FARNUM</th>
      <th>ANNNEWMEMNUM</th>
      <th>ANNREDMEMNUM</th>
      <th>EMPNUM</th>
      <th>EMPNUMSIGN</th>
      <th>BUSSTNAME</th>
      <th>COLGRANUM</th>
      <th>RETSOLNUM</th>
      <th>DISPERNUM</th>
      <th>UNENUM</th>
      <th>COLEMPLNUM</th>
      <th>RETEMPLNUM</th>
      <th>DISEMPLNUM</th>
      <th>UNEEMPLNUM</th>
      <th>WEBSITSIGN</th>
      <th>FORINVESTSIGN</th>
      <th>STOCKTRANSIGN</th>
      <th>PUBSTATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>2017.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>2018.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f000950527a6feb63ee1ce82bb22ddd1ab8b8fdffa3b91fb</td>
      <td>2017.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>开业</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f000950527a6feb63ee1ce82bb22ddd1ab8b8fdffa3b91fb</td>
      <td>2018.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>开业</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9c7fa510616a68309e4badf2a7a3123c0462fb85bf28ef17</td>
      <td>2017.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



对对象型数据进行特征编码


```python
# 删除缺失值大于0.5的特征
annual_clean = annual_report_info.dropna(thresh=annual_report_info.shape[0]*0.5, how='all', axis=1)
# 对对象型数据进行编码
annual_clean['BUSSTNAME'] = annual_clean['BUSSTNAME'].fillna('空')
ch = list(set(annual_clean['BUSSTNAME']))
en = ['kong', 'tingye', 'kaiye', 'xieye', 'jiesuan']
mapping = dict(zip(ch,en))
annual_clean['BUSSTNAME'] = annual_clean['BUSSTNAME'].map(mapping)
# one-hot编码
annual_clean = annual_clean.join(pd.get_dummies(annual_clean['BUSSTNAME'], prefix='BUSSTNAME'))
annual_clean.drop('BUSSTNAME', axis=1, inplace=True)
```

合并相同企业的数据，取平均值


```python
annual_clean = annual_clean.groupby('id',sort=False).agg('mean')
annual_clean = annual_clean.reset_index()
```

### tax_info

查看tax_info信息


```python
tax_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 29195 entries, 0 to 29194
    Data columns (total 9 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   id              29195 non-null  object 
     1   START_DATE      29195 non-null  object 
     2   END_DATE        29195 non-null  object 
     3   TAX_CATEGORIES  29195 non-null  object 
     4   TAX_ITEMS       29195 non-null  object 
     5   TAXATION_BASIS  25816 non-null  float64
     6   TAX_RATE        25816 non-null  float64
     7   DEDUCTION       24235 non-null  float64
     8   TAX_AMOUNT      29195 non-null  float64
    dtypes: float64(4), object(5)
    memory usage: 2.0+ MB
    


```python
tax_info.head()
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
      <th>START_DATE</th>
      <th>END_DATE</th>
      <th>TAX_CATEGORIES</th>
      <th>TAX_ITEMS</th>
      <th>TAXATION_BASIS</th>
      <th>TAX_RATE</th>
      <th>DEDUCTION</th>
      <th>TAX_AMOUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f000950527a6feb6c2f40c9d8477e73a439dfa0897830397</td>
      <td>2015/09/01</td>
      <td>2015/09/30</td>
      <td>印花税</td>
      <td>工伤保险（单位）</td>
      <td>72530.75</td>
      <td>0.0003</td>
      <td>-0.04</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f000950527a6feb6c2f40c9d8477e73a439dfa0897830397</td>
      <td>2015/09/01</td>
      <td>2015/09/30</td>
      <td>印花税</td>
      <td>失业保险（单位）</td>
      <td>72530.75</td>
      <td>0.0003</td>
      <td>-0.04</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f000950527a6feb6c2f40c9d8477e73a439dfa0897830397</td>
      <td>2015/09/01</td>
      <td>2015/09/30</td>
      <td>印花税</td>
      <td>医疗保险（单位）</td>
      <td>72530.75</td>
      <td>0.0003</td>
      <td>-0.04</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f000950527a6feb6c2f40c9d8477e73a439dfa0897830397</td>
      <td>2015/09/01</td>
      <td>2015/09/30</td>
      <td>印花税</td>
      <td>企业养老保险基金（单位）</td>
      <td>72530.75</td>
      <td>0.0003</td>
      <td>-0.04</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f000950527a6feb6c2f40c9d8477e73a439dfa0897830397</td>
      <td>2015/09/01</td>
      <td>2015/09/30</td>
      <td>印花税</td>
      <td>烟叶收购</td>
      <td>72530.75</td>
      <td>0.0003</td>
      <td>-0.04</td>
      <td>21.8</td>
    </tr>
  </tbody>
</table>
</div>



特征编码


```python
tax_clean = tax_info.copy()
# 对 START_DATE 和 END_DATE 进行处理
tax_clean['START_DATE'] = pd.to_datetime(tax_clean['START_DATE'],format='%Y-%m-%d')
tax_clean['END_DATE'] = pd.to_datetime(tax_clean['END_DATE'],format='%Y-%m-%d')
today = datetime.datetime.strptime('2020-10-13', '%Y-%m-%d')
tax_clean['PassedDay'] = tax_clean['START_DATE'].apply(lambda x: today-x).dt.days
tax_clean['TaxGapDay'] = (tax_clean['END_DATE'] - tax_clean['START_DATE']).dt.days
tax_clean.drop(['START_DATE','END_DATE'], axis=1, inplace=True)
# 对 TAX_CATEGORIES 和 TAX_ITEMS 编码
categories = list(set(tax_clean['TAX_CATEGORIES']))
items = list(set(tax_clean['TAX_ITEMS']))
tax_clean['TAX_CATEGORIES'] = tax_clean['TAX_CATEGORIES'].map(dict(zip(categories, list(range(len(categories))))))
tax_clean['TAX_ITEMS'] = tax_clean['TAX_ITEMS'].map(dict(zip(items,list(range(len(items))))))
```

将同公司的税务数据合并


```python
tax_clean = tax_clean.groupby('id',sort=False).agg('mean')
tax_clean = tax_clean.reset_index()
```

### change_info

查看change_info


```python
change_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45940 entries, 0 to 45939
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   id      45940 non-null  object 
     1   bgxmdm  45940 non-null  float64
     2   bgq     45940 non-null  object 
     3   bgh     45940 non-null  object 
     4   bgrq    45940 non-null  float64
    dtypes: float64(2), object(3)
    memory usage: 1.8+ MB
    


```python
change_info.head()
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
      <th>bgxmdm</th>
      <th>bgq</th>
      <th>bgh</th>
      <th>bgrq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>939.0</td>
      <td>9dec12da51cdb672a91b4a8ae0e0895f7bfeb243dfa3e0c8</td>
      <td>9dec12da51cdb672a91b4a8ae0e0895f4a56cbe3deca98...</td>
      <td>2.019060e+13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>112.0</td>
      <td>31487d8f256f16bd6244b7251be2ebb27b17bdfd95c8f3...</td>
      <td>31487d8f256f16bd6244b7251be2ebb27b17bdfd95c8f3...</td>
      <td>2.019060e+13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e9f7b28ec10e047000d16ab79e1b5e6da434a1697cce7818</td>
      <td>111.0</td>
      <td>54ca436ffb87f24c820178b45fcc3a7b</td>
      <td>f80e3376abcf81ad2a279d6d99046153</td>
      <td>2.017013e+13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e9f7b28ec10e047000d16ab79e1b5e6da434a1697cce7818</td>
      <td>128.0</td>
      <td>f1fdb1c866dc96638cbfb8b788b91393</td>
      <td>1eca8a0d8beca58d988f7dccab5dc868</td>
      <td>2.017013e+13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e9f7b28ec10e047000d16ab79e1b5e6da434a1697cce7818</td>
      <td>925.0</td>
      <td>54ca436ffb87f24c820178b45fcc3a7b</td>
      <td>f80e3376abcf81ad2a279d6d99046153</td>
      <td>2.017013e+13</td>
    </tr>
  </tbody>
</table>
</div>



bgq，bgh和bgrq给的信息不好处理，故直接删除这三个特征，只保留bgxmdm


```python
change_clean = change_info.drop(['bgq','bgh','bgrq'],axis=1)
# 合并数据
change_clean = change_clean.groupby('id',sort=False).agg('mean')
change_clean = change_clean.reset_index()
```

### news_info

查看news_info


```python
news_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10518 entries, 0 to 10517
    Data columns (total 3 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   id                10518 non-null  object
     1   positive_negtive  10518 non-null  object
     2   public_date       10518 non-null  object
    dtypes: object(3)
    memory usage: 246.6+ KB
    


```python
news_info.head()
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
      <th>positive_negtive</th>
      <th>public_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>f000950527a6feb62669d6a175fe6fdccd1eb4f7ca8e5016</td>
      <td>积极</td>
      <td>2016-12-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f000950527a6feb6e8bd9919e2ca363359bcfa997a0f9de7</td>
      <td>中立</td>
      <td>2017-08-09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f000950527a6feb6e8bd9919e2ca363359bcfa997a0f9de7</td>
      <td>消极</td>
      <td>2016-02-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d8071a739aa75a3bcf6fb0041ee883243251d30025ab9d45</td>
      <td>中立</td>
      <td>2018-06-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f000950527a6feb6d71de3382afa0bc5ff87bb65477f698a</td>
      <td>积极</td>
      <td>2015-06-29</td>
    </tr>
  </tbody>
</table>
</div>



进行特征编码


```python
news_clean = news_info.copy()
# 对 positive_negtive 进行one-hot编码
index = list(set(news_clean['positive_negtive']))
news_clean['positive_negtive'] = news_clean['positive_negtive'].map(dict(zip(index, ['neutral', 'positive', 'negtive'])))
news_clean = news_clean.join(pd.get_dummies(news_clean['positive_negtive']))
# 对 public_date进行处理
news_clean['public_date'] = pd.to_datetime(news_clean['public_date'],format='%Y-%m-%d',errors='coerce')
today = datetime.datetime.strptime('2020-10-13', '%Y-%m-%d')
news_clean['PublicDateDay'] = news_clean['public_date'].apply(lambda x: today-x).dt.days
# 因为含错误日期格式而导致的空值，所以对其进行填补
news_clean['PublicDateDay'] = news_clean['PublicDateDay'].fillna(news_clean['PublicDateDay'].median())
news_clean.drop(['positive_negtive', 'public_date'], axis=1, inplace=True)
# 同公司数据合并
news_clean = news_clean.groupby('id',sort=False).agg('mean')
news_clean = news_clean.reset_index()
```

### other_info

查看other_info


```python
other_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1890 entries, 0 to 1889
    Data columns (total 4 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   id                  1890 non-null   object 
     1   legal_judgment_num  1006 non-null   float64
     2   brand_num           909 non-null    float64
     3   patent_num          396 non-null    float64
    dtypes: float64(3), object(1)
    memory usage: 59.2+ KB
    

可以看到缺失值还是比较多的，就不考虑将该数据集融合了

### 合并数据集


```python
data_set = base_clean.merge(annual_clean, how='outer')
data_set = data_set.merge(tax_clean, how='outer')
data_set = data_set.merge(change_clean, how='outer')
data_set = data_set.merge(news_clean, how='outer')
data_set = data_set.fillna(-1)
```


```python
data_set.shape
```




    (24865, 73)



## 建模

分离训练集和验证集


```python
train_set = data_set.merge(entprise_info)
train_data = train_set.drop(['id','label'], axis=1)
train_label = train_set['label']
test_set = data_set.merge(entprise_evaluate)
test_data = test_set.drop(['id','score'], axis=1)
train_data.shape, test_data.shape
```




    ((14865, 72), (10000, 72))



定义评估函数，通过k折验证得到不同模型的分数，以便调参


```python
def evaluateModel(model,x,y):
    mean_f1=0
    folds=5
    sk = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2020)
    
    for trn_idx, val_idx in sk.split(x, y):
        trn_x, trn_y, val_x, val_y = x.iloc[trn_idx], y[trn_idx], x.iloc[val_idx], y[val_idx]

        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        mean_f1 += f1_score(val_y, val_pred)/sk.n_splits
        
    return mean_f1
```

简单训练随机森林，xgboost，lightgbm，catboost四个模型，得到对应的分数


```python
rf = RandomForestClassifier(oob_score=True, 
                            random_state=2020,
                            n_estimators= 95,
                            max_depth=12,
                            min_samples_split=7)
print('rf:',evaluateModel(rf,train_data,train_label))
```

    rf: 0.8281982187732465
    


```python
xlf = xgb.XGBClassifier(max_depth=8,
                      learning_rate=0.02,
                      n_estimators=75,
                      reg_alpha=0.005,
                      n_jobs=8,
                      importance_type='total_cover')
print('xlf:',evaluateModel(xlf,train_data,train_label))
```

    xlf: 0.8330400053008485
    


```python
llf = lgb.LGBMClassifier(num_leaves=12,
                       max_depth=6,
                       learning_rate=0.05,
                       n_estimators=85,
                       n_jobs=8)
print('llf:',evaluateModel(llf,train_data,train_label))  
```

    llf: 0.8385662194785999
    


```python
clf = cat.CatBoostClassifier(iterations=95,
                           learning_rate=0.05,
                           depth=8,
                           silent=True,
                           thread_count=8,
                           task_type='CPU')
print('clf:',evaluateModel(clf,train_data,train_label)) 
```

    clf: 0.840363023799186
    

采用加权平均的方法对四个模型进行模型融合


```python
mean_f1=0
folds=5
sk = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2020)
results = []

for idx, (trn_idx, val_idx) in enumerate(sk.split(train_data, train_label)):
    trn_x, trn_y, val_x, val_y = train_data.iloc[trn_idx], train_label[trn_idx], train_data.iloc[val_idx], train_label[val_idx]
    
    rf.fit(trn_x, trn_y)
    rf_pred = rf.predict(val_x)
    rf_prob = rf.predict_proba(val_x)
    rf_weight = f1_score(val_y, rf_pred)
    
    xlf.fit(trn_x, trn_y)
    xgb_pred = xlf.predict(val_x)
    xgb_prob = xlf.predict_proba(val_x)
    xgb_weight = f1_score(val_y, xgb_pred)

    llf.fit(trn_x, trn_y)
    lgb_pred = llf.predict(val_x)
    lgb_prob = llf.predict_proba(val_x)
    lgb_weight = f1_score(val_y, lgb_pred)

    clf.fit(trn_x, trn_y)
    cat_pred = clf.predict(val_x)
    cat_prob = clf.predict_proba(val_x)
    cat_weight = f1_score(val_y, cat_pred)
    
    #暴力搜索最佳权重
    weight = np.arange(0, 1.05, 0.1)
    maxscore = 0
    optweight = ()
    for i in weight:
        for j in weight[weight <= (1 - i)]:
            for k in weight[weight <= (1 - i - j)]:
                prob_weight = rf_prob*i + xgb_prob*j + lgb_prob*k + cat_prob*(1 - i - j - k)
                score = f1_score(val_y, np.argmax(prob_weight,axis=1))
                if score > maxscore:
                    maxscore = score
                    optweight = (i, j, k, 1-i-j-k)
    print('第{}次验证f1_score：{}'.format(idx + 1, maxscore))
    print('权重为rf, xgb, lgb, cat：', optweight)
    mean_f1+=maxscore/sk.n_splits
    
    test_rf = rf.predict_proba(test_data)
    test_xgb = xlf.predict_proba(test_data)
    test_lgb = llf.predict_proba(test_data)
    test_cat = clf.predict_proba(test_data)
    test_pred = test_rf*optweight[0] + test_xgb*optweight[1] + test_lgb*optweight[2] + test_cat*optweight[3]
    results.append(test_pred)
print('线上验证f1_score: ', mean_f1)
```

    第1次验证f1_score：0.8541666666666667
    权重为rf, xgb, lgb, cat： (0.5, 0.2, 0.0, 0.3)
    第2次验证f1_score：0.8564231738035265
    权重为rf, xgb, lgb, cat： (0.1, 0.5, 0.0, 0.4)
    第3次验证f1_score：0.8249400479616307
    权重为rf, xgb, lgb, cat： (0.5, 0.0, 0.5, 0.0)
    第4次验证f1_score：0.8676470588235294
    权重为rf, xgb, lgb, cat： (0.2, 0.1, 0.4, 0.30000000000000004)
    第5次验证f1_score：0.8478802992518704
    权重为rf, xgb, lgb, cat： (0.0, 0.30000000000000004, 0.4, 0.29999999999999993)
    线上验证f1_score:  0.8502114493014447
    


```python
result = (sum(results)/sk.n_splits)[:,1]
submit_file = pd.DataFrame({'id': test_set['id'], 'score': result.tolist()})
submit_file
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
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9c7fa510616a683058ce97d0bc768a621cd85ab1e87da2a3</td>
      <td>0.025312</td>
    </tr>
    <tr>
      <th>1</th>
      <td>da8691b210adb3f67820f5e0c87b337d63112cee52211888</td>
      <td>0.025154</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9c7fa510616a68309e4badf2a7a3123c0462fb85bf28ef17</td>
      <td>0.025184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f000950527a6feb6ed308bc4c7ae11276eab86480f8e03db</td>
      <td>0.028397</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f000950527a6feb617e8d6ca7025dcf9d765429969122069</td>
      <td>0.026591</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>f1c1045b13d18329a2bd99d2a7e2227688c0d69bf1d1e325</td>
      <td>0.056096</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>f000950527a6feb6bde38216d7cbbf32e66d3a3a96d4dbda</td>
      <td>0.528226</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>da8691b210adb3f65b43370d3a362f4aa1d3b16b5ba0c9d7</td>
      <td>0.030066</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>516ab81418ed215dcbbf0614a7b929e691f8eed153d7bb31</td>
      <td>0.071862</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>9c7fa510616a68303d3427d4bfd4b0cf3e4843f2bf3f637a</td>
      <td>0.036229</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>




```python
submit_file.to_csv('submit01.csv', index=0)
```
