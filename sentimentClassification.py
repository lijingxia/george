import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
# 加载数训练集和测试数据集
print('Loading data...')
train = pd.read_csv('dataSet/train.csv')
test = pd.read_csv('dataSet/test.csv')
# 打印维度
# print(train.shape)
# print(test.shape)
# 打印训练集
# print(train.head(10))
# 打印测试集
# print(test.head(10))
# 缺省值分析
# print(train.isnull().sum())
# print(test.isnull().sum())
from keras.preprocessing.text import text_to_word_sequence
'''[George]
    text_to_word_sequence 这个模块可以将文本转换成小写，
    并且自动用空格对文本进行分割
    它并不会移除反斜杠，所以我们需要将反斜杠添加到过滤器里
    下面将训练集中的所有标点符号去除，用空格作为分隔符提取单词，
    并且全部转换为小写
'''
# print(train.head())
train['comment_text_words'] = train.comment_text.apply(text_to_word_sequence, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')
# print(train.head())
# print(train.iloc[0, :]['comment_text'])
# print(train.iloc[0, :]['comment_text_words'])
# print(train.head())
test['comment_text_words'] = test.comment_text.apply(text_to_word_sequence, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')

# 将预训练好的 embeddings 进行处理，返回几个有用的变量：
# words_to_index：将单词和编号进行映射
# index_to_words：将编号和单词进行映射，与 words_to_index 相反
# word_to_vec_map：将单词与它的 embedding 进行映射
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
#         # 将单词保存到一个集合当中
        words = set()
#         # 将单词与 embedding 的映射保存到一个字典中
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
#             # 列表的第一个元素是单词
            curr_word = line[0]
#             # 将单词加入到集合中
            words.add(curr_word)
#             # 列表的其余元素是 embedding，将单词与 embedding 进行映射，然后保存到字典中
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
#         # 将单词进行编号，编号从 1 开始
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            # 创建映射，key 是单词，value 是编号
            words_to_index[w] = i
            # 创建映射，key 是编号，value 是单词
            index_to_words[i] = w
            # 编号递增
            i = i + 1
            
#     # 将生成的 3 个映射返回
    return words_to_index, index_to_words, word_to_vec_map


# # glove.6B.50d.txt 是网上已经预训练好的 word embedding 文件
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('dataSet/glove.6B.50d.txt')

def sentences_to_indices(X):
    X_indices = []
    for word in X:
        try:
            X_indices.append(word_to_index[word])
        except:
            pass
    
    return X_indices

train['comment_text_indexes'] = train.comment_text_words.apply(sentences_to_indices)
test['comment_text_indexes'] = test.comment_text_words.apply(sentences_to_indices)

print(train.head())

comment_text_max_words_length = np.max([np.max(train.comment_text_words.apply(lambda x: len(x)))
                                   , np.max(test.comment_text_words.apply(lambda x: len(x)))])
print("最大文本长度是 "+str(comment_text_max_words_length))


# 查看商品名的长度的分布
train.comment_text_words.apply(lambda x: len(x)).hist()

# 从图中可以发现，文本长度集中在0-400之间，所以我们把最大文本长度设置为200
MAX_COMMENT_TEXT_SEQ = 200

# pad_sequences 函数用于将数字列表进行填充
# 如果列表长度大于最大长度，那么将列表进行裁剪，如果列表长度小于最大长度，那么将列表补充到最大长度，并且默认填充0

from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {'comment_text': pad_sequences(dataset.comment_text_indexes, maxlen=MAX_COMMENT_TEXT_SEQ)}
    return X

# 将训练集数据的文本编号列表进行填充，并且提取出来
X_train = get_keras_data(train)
# 将测试集数据的文本编号列表进行填充，并且提取出来
X_test = get_keras_data(test)


import pickle

datafile = open('data.pkl', 'wb')

pickle.dump(X_train, datafile)
pickle.dump(X_test, datafile)
pickle.dump(word_to_index, datafile)
pickle.dump(index_to_word, datafile)
pickle.dump(word_to_vec_map, datafile)

datafile.close()
