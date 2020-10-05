# 第二周周报

## 学习进度
- 完善数据分析部分，将特征根据数据类型区分开
- 开始进行数据预处理，包括：
    - 几种缺失值的填充方法：用0、平均值、众数或上方的值来填充
    - 特征类型的转化：将非数值型转化为数值型
- 大致认识了几种好用的模型：GBDT，XGBoost，LightGBM，但详细的还未做深入理解
- 用LightGBM模型运行经过上述处理的数据集，下为提交记录和排名：  
![17343118的提交](https://img-blog.csdnimg.cn/20201005124422905.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Rpb3NtYWlfa2luZ3Nv,size_16,color_FFFFFF,t_70#pic_center)
![17343118的排名](https://img-blog.csdnimg.cn/20201005124422812.png#pic_center)

## 比赛报告
从第一周完到第二周完之间的部分为本周的工作，我的运行代码另存在其他笔记中  
https://github.com/AmarKingso/DataMiningTraining/blob/master/FinancialRiskControl/FinancialRisk.md  

## 学习心得
本周的学习主要还是围绕着数据预处理这一块，模型方面只做了一个简单的了解，还未做较深入的了解。  
在学习预处理的同时，也学习了很多pandas、DataFrame相关的实用的函数，包括给离散的类别做one-hot的映射，用到的get_dummies函数。然后就是一些类型的转化上，要考虑其原来类型是否具有大小关系，例如这里面的grade和subGrade，以字母代表等级，都是有大小关系的，而我一开始使用one-hot编码的方式，得到的结果不是很理想，换成了数字映射后就好很多。数据预处理这一块内容还有很多，包括我还没有做的异常值处理，和数据分箱等等，都还在了解阶段，留待之后的学习中逐步完善。
