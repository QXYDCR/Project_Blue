import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# 读取精简版数据
path = "C:/Users/Chihiro/Desktop/RL Paper/Project/Project_Blue_1/training.1600000.processed.noemoticon.csv"

# dataset = pd.read_csv(path, engine ="python", header = None)

# dataset = pd.read_csv("C:/Users/Chihiro/Desktop/RL Paper/Project/Project_Blue_1"
#                       "/testdata.manual.2009.06.14.csv",
#                       engine ="python", header = None)

# (498, 6)
# print(dataset.shape)

# 数据表信息
# print(dataset.info())

# 默认显示前5行)
#    0  1  ...         4                                                  5
# 0  4  3  ...    tpryan  @stellargirl I loooooooovvvvvveee my Kindle2. ...
# 1  4  4  ...    vcu451  Reading my kindle2...  Love it... Lee childs i...
# 2  4  5  ...    chadfu  Ok, first assesment of the #kindle2 ...it fuck...
# 3  4  6  ...     SIX15  @kenburbary You'll love your Kindle2. I've had...
# 4  4  7  ...  yamarama  @mikefish  Fair enough. But i have the Kindle2...
# print(dataset.head())

# 统计各个类别数据占比
# value_counts --> Return a Series containing counts of unique values.
# dataset[0], 返回第0列的元素，即是[N, 1]
# print(dataset[0].value_counts())

# 类型转换 --> 分类变量
# 创建一个新的列，将类型改变为 'category'
# dataset['sentiment_category'] = dataset[0].astype('category')
# # 结果和dataset[0]一样
# print(dataset['sentiment_category'].value_counts())

# 分类变量值转换为 0 和 1 两个类别
# dataset['sentiment'] = dataset['sentiment_category'].cat.codes
# print(dataset.head())
# 统计类别占比
# print(dataset['sentiment'].value_counts())

# 保存文件
# dataset.to_csv("C:/Users/Chihiro/Desktop/RL Paper/Project/Project_Blue_1"
#                       "/training-processed.csv", header=None, index=None)





























