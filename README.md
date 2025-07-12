# EnsemGlyPred

EnsemGlyPred: 融合深层语义特征与序列信息的多层级赖氨酸糖化位点预测框架

本研究构建了基于多特征融合与加权集成学习的赖氨酸糖化位点预测框架，通过整合传统序列特征与先进蛋白质语言模型表征，显著提高了预测的准确性和可靠性。

## 文件说明

**code files文件夹**: 包含模型训练和测试的核心代码文件。
- **Train文件夹**: 包含三个基础模型的训练代码，分别基于AAC特征、PAAC特征和ProGen2特征构建XGBoost和BiLSTM分类器，以及集成训练代码。
- **Test文件夹**: 包含对应的模型测试和集成预测相关代码。

**dataset文件夹**: 包含经过严格去冗余处理的训练数据集(7146个样本)和独立测试数据集(400个样本)，正负样本保持平衡。

**feature_npy文件夹**: 存储提取的三类特征向量，包括AAC特征(20维)、PAAC特征(25维)和经PCA降维的ProGen2特征(300维)，以及对应的标签文件。

**model files文件夹**: 包含训练好的基础模型文件。由于ProGen2预训练模型文件过大，无法直接上传到GitHub，请从HuggingFace下载：https://huggingface.co/hugohrban/progen2-base 。由于大型模型参数的随机性，请确保设置随机种子，本研究的随机种子设置为58。

## 模型架构

本研究构建了基于多特征融合的加权集成分类框架，包括三个基础模型： AAC-XGBoost模型: 基于氨基酸组成特征的XGBoost分类器；PAAC-XGBoost模型: 基于伪氨基酸组成特征的XGBoost分类器；ProGen2-BiLSTM模型: 基于蛋白质语言模型特征的BiLSTM分类器

集成学习策略采用加权软投票，权重分配为： AAC-XGBoost: 0.33； PAAC-XGBoost: 0.40 ；ProGen2-BiLSTM: 0.27

## 联系方式

如有问题请联系：
- 邮箱：1191230413@stu.jiangnan.edu.cn