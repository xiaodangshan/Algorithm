# Griewank函数

import matplotlib.pyplot as plt
import math
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


class GriewankFunction:

    def griewank_function(self, param1, param2):
        itemOne = 0
        itmeTwo = 0
        for i in range(0, param1):
            itemOne += (param1 ** 2) / 4000
            itmeTwo *= (math.cos(param2 / math.sqrt(i + 1)))
        return 1 + itemOne - itmeTwo

    def show_griewank_function(self, bounds):
        ax = plt.axes(projection='3d')
        xx = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
        yy = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
        X, Y = np.meshgrid(xx, yy)
        Z = 1 + (X ** 2 + Y ** 2) / 4000 - (np.cos(X / 1) * np.cos(Y / np.sqrt(2)))
        minZ = np.min(Z)
        indexMin = np.argmin(Z)
        ax.plot_surface(X, Y, Z, cmap='rainbow')
        plt.title("GriewankFunction,minValue%s" % round(minZ, 2))
        plt.show()


if __name__ == '__main__':
    function = GriewankFunction()
    function.show_griewank_function([-10, 10])
