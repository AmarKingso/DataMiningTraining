# 第八周周报

## 进度
- 大致了解了八个数据集中的信息，发现除了base_info以外的数据集企业信息缺失严重，只有一小部分企业记录在案
- 目前只使用了一个base_info进行建模，简单进行了预处理(特征编码、缺失值处理)
- 采用手动调参的方法，对比四个模型（随机森林、xgboost、lightgbm、catboost）的结果，采用加权平均法进行模型融合，最优权重通过暴力搜索获得。模型融合后的结果比单模有显著提高，这在我的notebook中有较为直观的体现。
- 目前代码存在很大改进空间，个人觉得还未完善的方面有：融合其他数据集特征，更加合适的特征编码，参数上的优化；总的来说还是特征工程有大量工作留待完成。
- 提交成绩：
![17343118的提交](https://img-blog.csdnimg.cn/20201116170100945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Rpb3NtYWlfa2luZ3Nv,size_16,color_FFFFFF,t_70#pic_center)

## 比赛报告
