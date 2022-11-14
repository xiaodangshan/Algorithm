# 因为有多个类别，所以
# 这是多分类（multiclass classification）问题的一个例子。因为每个数据点只能划分到一个类别，
# 所以更具体地说，这是单标签、多分类（single-label, multiclass classification）问题的一个例子。
# 如果每个数据点可以划分到多个类别（主题），那它就是一个多标签、多分类（multilabel, multiclass classification）问题。

# 8982 个训练样本和 2246 个测试样本,数据限定为前 10 000 个
from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 对于这个例子，最好的损失函数是 categorical_crossentropy（分类交叉熵）。它用于
# 衡量两个概率分布之间的距离，这里两个概率分布分别是网络输出的概率分布和标签的真实分
# 布。通过将这两个分布的距离最小化，训练网络可使输出结果尽可能接近真实标签。
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在新数据上生成预测结果
predictions = model.predict(x_test)
predictions[0].shape
np.sum(predictions[0])
# 最大的元素就是预测类别，即概率最大的类别。
np.argmax(predictions[0])