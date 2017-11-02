基于卷积神经网络参数优化的情感分析Code
本github上的代码是“基于卷积神经网络参数优化的中文情感分析”论文所编写


## Requirement 	
- python 3 	
- tensorflow >0.8 	
- numpy 	
- scikit-learn 	
- jieba 	

## DataSets	
实验的原始数据存放在data_origin文件夹下，分为酒店评论数据和电商评论数据，其中每个文件一条评论  
实验的预处理数据存放在data_process文件夹下，分为酒店评论数据和电商评论数据，每种数据又分为消极评论和积极评论，并将原始评论数据整合到
一个文件中，每行一条评论数据。

## 数据预处理类——preprocessing.py	
文件中定义了各类方法，方便对原始数据进行预处理

## 数据加载类——data_loader.py
文件中定义了训练数据时，可以使用的加载数据的方法

## 卷积神经网络架构模型——cnn_graph.py
	
## 模型训练
- train_n.py      模型训练，但不会产生summary和checkout，使用10折交叉验证生成准确率
- train_y.py      模型训练，产生summary和checkout，使用10折交叉验证生成准确率,训练后可以通过tensorflow查看模型的训练趋势图，但训练速度较慢
	
模型训练时，请通过修改文件中模型的超参来查看模型对参数的敏感度，其中包括：
- embedding_dim
- filter_sizes
- num_filters
- dropout_keep_prob 
- l2_reg_lambda
- batch_size
- num_epochs	
以及词向量的地址和训练数据地址
- w2v_path
- file_dir

## 词向量模型
提前训练的词向量由于文件过大，不易上传，故存放在百度云盘中
地址：http://pan.baidu.com/s/1dFzBLPv  提取码：ficp
云盘中包括：
- retrain_vectors_50      包含酒店评论数据的词向量，维度50
- retrain_vectors_100      包含酒店评论数据的词向量，维度100
- retrain_vectors_200      包含酒店评论数据的词向量，维度200
- vectors_50      搜狗新闻数据集训练的词向量，维度50
- vectors_100      搜狗新闻数据集训练的词向量，维度100
- vectors_200      搜狗新闻数据集训练的词向量，维度200
- vectorsPlus_50      搜狗新闻数据集+wiki数据训练的词向量，维度50
- vectorsPlus_100      搜狗新闻数据集+wiki数据训练的词向量，维度100
- vectorsPlus_200      搜狗新闻数据集+wiki数据训练的词向量，维度200
- wiki_vectors_50      wiki数据训练的词向量，维度50
