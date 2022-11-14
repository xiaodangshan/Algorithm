from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

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
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 使用训练好的网络在新数据上生成预测结果
predict = model.predict(x_test)
print(predict)
