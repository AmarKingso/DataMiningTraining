# 第七周周报

## 项目总结
这周是第一次比赛的最后一周，也没有再去研究如何提高分数。所以这周想对前几周所学知识做一个总结。通过参阅官方提供的学习资料和TA的介绍，我将整个数据挖掘项目的过程分为数据分析（EDA）、数据预处理和建模。
- 数据分析  
作为整个项目的第一步，我们要做的当然是对数据集有一个全面的了解，例如了解各个特征表示的含义，在该领域上的重要程度；亦或是着眼于数据本身，样本类型、数量有多少，各特征的缺失率以及数据类型如何，总体分布又如何。当我们对数据总体有一个大致的探索后，便能够相对轻松的开始之后的工作。
- 数据预处理  
这一部分是整个数据挖掘过程中需要投入精力最多的部分，因为一组干净的数据才能尽可能的保证最后的结果不受到外界因素的影响。我们可以通过数据分箱来直接对整个数据进行处理，也可以通过选择合适的方法，去填补缺失值（平均值、中位数、删除）、处理异常值（平均值、中位数、删除）、校对数据冗余和不一致，以保证数据的干净；然后在特征上，对特征进行类型转化、将数据归一化（最小-最大、z-score）、合并创建新的特征或是降维（特征选择（pearson相关系数、最大信息系数、距离相关系数）、正则化），以提高我们在运行模型时的效率和和最终结果的准确率与泛化能力。一个好的数据预处理往往决定了最后得到的结果。
- 建模  
这一步需要操心的事并不多，我们在前期的大量工作都是为此打基础。我们此需要的工作就只剩下建模、调参以及模型融合。  
模型的选择上，需要我们自己去运行数据得到结果，再相互比较，虽然比较笨，但却很有效。可供选择的模型有很多，从经典的svm、逻辑回归，到决策树模型、基于bagging（随机森林）和boosting（GBDT、XGBoost、LightGBM）的集成模型，以及各类神经网络。调参上则可以通过自动化调参得到最优参数，了解到的方法有：贪心调参、网格搜索和贝叶斯调参；而模型融合部分有简单的加权平均法、加权投票法、排序融合等，也有stacking、blending等相对复杂的算法。