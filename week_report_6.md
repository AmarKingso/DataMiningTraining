# 第六周周报

## 学习进度
- 了解了几种自动化调参方式：
    - 贪心调参
    - 网格搜索
    - 贝叶斯优化
- 深入了解贝叶斯调参的实现，因为其是不断添加样本点去更新目标函数的后验分布，所以叫做贝叶斯调参
- 实现贝叶斯调参，并用所得的参数建模
- 提交记录和排名：  
![17343118的提交](https://img-blog.csdnimg.cn/2020110300015558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Rpb3NtYWlfa2luZ3Nv,size_16,color_FFFFFF,t_70#pic_center)

## 比赛报告
这周的理论学习依旧是放在之前的报告中，而代码实现则是在建模的报告中：  
[理论学习](https://github.com/AmarKingso/DataMiningTraining/blob/master/FinancialRiskControl/FinancialRisk.md)  
[代码实现](https://github.com/AmarKingso/DataMiningTraining/blob/master/FinancialRiskControl/Modeling.md)

## 学习心得
这周主要做的就是研究自动化调参，而研究的三种调参方法中，贝叶斯无疑是最适合该数据集的，不论是从样本空间的大小，还是从得到的参数结果上来说。但不得不说，即使解决了模型调参的问题，我们依然需要对贝叶斯优化本身进行调参，例如贝叶斯优化的迭代次数，目标函数中建模的一些参数，被调参的参数的范围等等，甚至需要一些运气，比如设置贝叶斯优化的seed。我第一次调参完运行模型，结果反而没有未调参之前好，个人认为是参数范围设置，以及修改了原本建模的一些参数所致。到这里整个项目就完成的差不多了，在整个实现过程中，我确实学到了五花八门的数据挖掘方法，同时也体会到了数据清洗的工作量之大以及难度。希望能把自己这次比赛学到的知识，更好地运用到之后的比赛中。