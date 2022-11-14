# from keras.datasets import mnist
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.array(12)
# ndim = x.ndim
# print(ndim)
# # 载入数据集
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
# #张量切片
# my_slice = train_images[10:100]
# print(my_slice.shape)
#
# #等同于下面这个更复杂的写法，给出了切片沿着每个张量轴的起始索引和结束索引。: 等同于选择整个轴
# my_slice = train_images[10:100, :, :]
# my_slice.shape
# #等同于下面的写法
# my_slice = train_images[10:100, 0:28, 0:28]
# my_slice.shape
#
# #可以沿着每个张量轴在任意两个索引之间进行选择。例如，你可以在所有图像的右下角选出 14 像素×14 像素的区域
# my_slice = train_images[:, 14:, 14:]
# #　数据批量，对于这种批量张量，第一个轴（0 轴）叫作批量轴（batch axis）或批量维度
# batch = train_images[:128]
#
# #数组中的数字和5比较，取最大值
# A = np.array([[1,8,3,6,5],[9,2,7,4,5]])
# np.maximum(A, 5)
#
# #张量变形
# x = x.reshape((6, 1))0
#
# #创建一个形状为 (300, 20) 的零矩阵
# x = np.zeros((300, 20))
# #张量转置
# x = np.transpose(x)
import numpy as np

results = np.zeros((1, 10))
results[0,[0]]=1
print(results)