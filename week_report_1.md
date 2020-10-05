# 第一周周报

## 学习进度
- 观看比赛提供的部分学习资料，对完整的数据挖掘项目过程有了一定的了解，主要分为四个部分：
    - 数据eda
    - 特征工程
    - 建模调参
    - 模型融合
- 大致了解了本次比赛的片评价方法——AUC的信息：
    - 假正例（FP）：若一个实例是负类，而被预测为正类
    - 真正例（TP）：若一个实例是正类，而被预测为正类
    - ROC曲线： x轴为假正例率，y轴为真正例率
    - AUC：ROC曲线与坐标轴围成图形的面积，其介于0.5和1.0之间，越接近1.0时真实性越高，反之越低
- notebook在数据分析上十分便捷，故使用来进行本次比赛
- 学习了python的一些数据可视化的方法
- 提取了数据集的整体上的一些信息，包括样本数量，特征维度、缺失值、唯一值和数据类型

## 比赛报告
我会将此次比赛的全过程整理成一份报告，并且会根据每周周报上交的时间，来表明该报告在每一周的进度。该报告网址为：
https://github.com/AmarKingso/DataMiningTraining/blob/master/FinancialRiskControl/FinancialRisk.md

## 学习心得
工具方面：以前使用python都是用vscode执行一些比较轻量的任务，这一次比赛是我第一次尝试使用notebook来进行数据分析处理这些工作，能够边写代码边记录想法，确实要方便很多，并且每次运行不用重新之前的代码，能够省下很多的时间。
理论方面：数据挖掘真的有很多需要学习的知识，不光是数据的预处理，还有处理前需要的eda，要知道提取哪些信息才是有用的，还要考虑特征间的关系，寻找其隐含的信息，对问题进行建模。但我还是希望能够通过自己之后的学习，掌握到更多的方法，并将其应用到比赛中，在排名的提升中收获成就感。