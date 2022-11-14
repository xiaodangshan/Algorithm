from keras.datasets import boston_housing
from keras import models
from keras import layers

# 404 个训练样本和 102 个测试样本,每个样本都有 13 个数值特征，比如人均犯罪率、每个住宅的平均房间数、高速公路可达性等。
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# 对于将取值范围差异很大的数据,普遍采用的最佳实践是对每个特征做标准化，即对于输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除
# 以标准差，这样得到的特征平均值为 0，标准差为 1。用 Numpy 可以很容易实现标准化。
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
