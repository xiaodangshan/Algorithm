from keras.datasets import imdb
import numpy as np

# 参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词。
# train_data 和 test_data 这两个变量都是评论组成的列表，每条评论又是单词索引组成的列表（表示一系列单词）。train_labels 和 test_labels 都是 0 和 1 组成的列表，其中 0
# 代表负面（negative），1 代表正面（positive）
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# print(train_data[0])
# print(train_labels[0])
# print(len(train_data[0]))
# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

# for sequence in train_data[0:1]:
#   for i, sequence in enumerate(sequence):
#       print(i,sequence)


# >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# >>> list(enumerate(seasons))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        print(i, sequence)
        results[i, sequence] = 1
        print(results)
    return results


x_train = vectorize_sequences(train_data[0:2])
print(x_train[0])
