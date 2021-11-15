# encoding:utf-8
# author:FuJun WANG
from functools import reduce
import math
import random
import copy
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

from mpl_toolkits.mplot3d import Axes3D
import numpy as np


##默认所有的函数都是求最小值问题
class testPro:
    def __init__(self):
        self.Pro1 = None

    def GriewankFunction(self, x):
        itemOne = 0
        itmeTwo = 0
        for i, value in enumerate(x):
            itemOne += (value ** 2) / 4000
            itmeTwo *= (math.cos(value / math.sqrt(i + 1)))
        return 1 + itemOne - itmeTwo

    def GeneralizeRastrigin(self, x):
        s = 0
        for value in x:
            s += (value ** 2 - 10 * math.cos(2 * math.pi * value) + 10)
        return -s

    def AckleyFunction(self, x):
        s1 = 0
        s2 = 0
        for value in x:
            s1 += value ** 2
            s2 += math.cos(2 * math.pi * value)
        s1 = s1 / len(x)
        s2 = s2 / len(x)
        return -20 * math.exp(-0.2 * math.sqrt(s1)) - math.exp(s2) + 20 + math.e

    def showGriewankFunction(self, n, bounds):  ##可视化函数
        """
        :param n: 可视化几阶次的函数，一般取2或3（因为最高我们只能可视化三维）
        :param bounds: 函数的边界
        :return:
        """
        if n == 2:
            fig = plt.figure()
            value = [self.GriewankFunction(x) for x in np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 1000)]
            plt.plot(range(len(value)), value)
            plt.title("GriewankFunction")
            plt.xlabel("x")
            plt.ylabel("value")
        elif n == 3:
            fig = plt.figure()
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

    def showGeneralizeRastrigin(self, n, bounds):
        if n == 2:
            pass
        if n == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            xx = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
            yy = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
            X, Y = np.meshgrid(xx, yy)
            Z = -(X ** 2 - 10 * np.cos(2 * math.pi * X) + 10) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y) + 10)
            minZ = np.min(Z)
            ax.plot_surface(X, Y, Z, cmap='rainbow')
            plt.title("GeneralizeRastrigin,minValue%s" % round(minZ, 2))
            plt.show()

    def showAckleyFunction(self, n, bounds):
        if n == 2:
            pass
        if n == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            xx = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
            yy = np.arange(bounds[0], bounds[1], (bounds[1] - bounds[0]) / 2000)
            X, Y = np.meshgrid(xx, yy)
            Z = -20 * np.exp(-0.2 * np.sqrt((X ** 2 + Y ** 2) / 2)) - np.exp(
                (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)) / 2) + 20 + math.e
            minZ = np.min(Z)
            ax.plot_surface(X, Y, Z, cmap='rainbow')
            plt.title("AckleyFunction,minValue%s" % round(minZ, 2))
            plt.show()


class fitnessFunc:
    def fitneForMin(self, value, average):
        # math.exp(-(value / (average)))
        if value >= 0:
            return 1 / (1 + value)
        else:
            return 1 + math.fabs(value)

    def fitneForMax(self, value, average):
        return math.exp(value / average)


class Nectar:  ##蜜源
    def __init__(self, pos):
        # self.pro=pro##蜜源对应的评价函数
        self.LIMIT = 8
        self.pos = pos  ##蜜源位置
        self.beAbandoned = 0  ##被丢弃的次数
        self.beChoosed = 1  # 被选择的
        self.followers = []
        self.func = testPro().GriewankFunction
        self.fitFunc = fitnessFunc().fitneForMin

    def getValue(self):  ##该位置在所给评价标准为pro的情况下的值，即蜜源大小
        return self.func(self.pos)

    def getBeAbandoned(self):
        return self.beAbandoned

    def addAbandoned(self):
        self.beAbandoned += 1

    def addBechoosed(self):
        self.beChoosed += 1

    def getPos(self):
        return self.pos

    def setPos(self, pos):
        self.pos = pos

    def getFollowers(self):
        return self.followers

    def setFollowers(self, newFoll):
        self.followers = newFoll

    def addFolloers(self, f):  ##该蜜源增加的采蜜者
        self.followers.append(f)

    def getLimit(self):
        return self.LIMIT


class Bee:
    def __init__(self, id, type):
        self.id = id
        self.type = type  ##type=1表示是雇佣蜂2表示是追随蜂3表示的是侦察蜂

    def setType(self):
        self.type = type

    def getType(self):
        return self.type

    def getId(self):
        return self.id


class ABC:
    def __init__(self, pro, B, E, L, S, I, bound):
        self.pro = pro  # 问题
        self.B = B  # 蜂群的大小
        self.E = E  # 雇佣蜂的数目
        self.L = L  # 追随蜂的数目
        self.S = S  # 侦察峰的数目
        self.I = I  # 最大的迭代次数
        self.st = []  # 保存蜜源
        self.bound = bound
        self.onLookList = []  ##追随蜂列表
        self.scoutsList = []  # 侦察蜂列表
        self.bestValue = None  # 最优值
        self.bestPos = None  # 最优值位置
        self.fitnessFun = fitnessFunc().fitneForMin
        self.resultRecord = []  ##存放中间最优值
        self.resultForPicture = []
        self.pictureIndex = [0, 10, 20, 30, 50, 70, 90, int(100 + 0 * (self.I - 100) / 5),
                             int(100 + 1 * (self.I - 100) / 5), int(100 + 2 * (self.I - 100) / 5),
                             int(100 + 3 * (self.I - 100) / 5), int(100 + 4 * (self.I - 100) / 5)]  # 在第几次循环中拍快照

    ##用二阶的grienWankFunc进行测试
    def getFeasibleSol(slef, bounds):  ##随机产生点
        return bounds[0] + random.random() * (bounds[1] - bounds[0])

    def initial(self):
        # 初始化蜜源
        for i in range(self.B):
            if i < self.E:  ##前E个是雇佣蜂,雇佣蜂的数量与
                # thisBee=Bee(i,1)##初始化一只小蜜蜂
                randomPos = [self.getFeasibleSol(self.bound), self.getFeasibleSol(self.bound)]
                thisNectar = Nectar(randomPos)  ##初始化蜜源的位置
                thisNectar.addFolloers(i)  ##将这只雇佣蜂添加到这个蜂源上
                self.st.append(thisNectar)
            else:
                ##这些是追随峰
                # thisBee=Bee(i,2)
                self.onLookList.append(i)
        for i in range(self.S):  ##侦察蜂
            # thisBee=Bee(i,3)
            self.scoutsList.append(i)
        bestIndex, self.bestValue = list(min(enumerate([item.getValue() for item in self.st])))
        self.bestPos = self.st[bestIndex].getPos()  # 目前最优的函数值，以及最优值对应的点

    def getRandomPos(self, pos1, pos2):
        randomStep = random.random()
        return [pos1[0] + randomStep * (pos2[0] - pos1[0]), pos1[1] + randomStep * (pos2[1] - pos1[1])]

    def roulette(self, pList):
        arrow = random.random()
        s = 0
        for index, value in enumerate(pList):
            s += value
            if s >= arrow:
                return index

    def employedPro(self):
        newStList = []  # 新的蜂源集合
        for nc in self.st:
            if len(nc.getFollowers()) == 0:
                pass  # 如果这个蜜源没有一个开采这者了，就废弃
            else:
                flyAway = []
                notFly = []
                for bee in nc.getFollowers():
                    if random.random() < 0.3:
                        self.onLookList.append(bee)
                        continue  # 成为跟随峰
                    if random.random() < 0.6 and random.random() > 0.3:  # 成为侦察蜂
                        self.scoutsList.append(bee)
                        continue
                    else:
                        # 探索一个新的位置,先随机的从现有的st中随机的选择一个元素
                        ranNc = random.choice(self.st)
                        newPos = self.getRandomPos(nc.getPos(), ranNc.getPos())
                        newNc = Nectar(newPos)
                        ##贪婪求法,
                        if newNc.getValue() < nc.getValue():  # 如果新探索的位置好，就放弃现在的位置

                            newNc.addFolloers(bee)
                            newStList.append(newNc)
                            flyAway.append(bee)  ##这只小蜜蜂飞走了
                            nc.addAbandoned()  ##被丢弃的个数加1
                            if newNc.getValue() < self.bestValue:  # 是否能信最优值
                                self.bestValue = newNc.getValue()
                                self.bestPos = newNc.getPos()
                                print("目前最优", self.bestValue, "|", self.bestPos)
                        else:
                            nc.addBechoosed()  # 坚持现在的,当前位置被宠幸加1
                            notFly.append(bee)
                ##所有蜜蜂遍历结束看当前还剩几个蜜蜂
                if len(flyAway) == len(nc.getFollowers()):  # 当前蜜源的蜜蜂飞走完了，该位置被废弃
                    pass
                elif (nc.getBeAbandoned() >= nc.getLimit()):  ##如果当前位置被丢弃过多,放弃该蜜源，此地的小蜜蜂成为其他峰
                    for bee in nc.getFollowers():
                        if random.random() < 0.5:  ##等概率的成为追随峰或者侦察峰
                            self.onLookList.append(bee)
                        else:
                            self.scoutsList.append(bee)
                else:
                    nc.setFollowers(notFly)
                    newStList.append(nc)
        self.st = copy.deepcopy(newStList)

    def onLookerPro(self):
        valueList = []
        for nc in self.st:
            valueList.append(nc.getValue())
        aveValue = sum(valueList) / len(valueList)
        fitnessList = [self.fitnessFun(item, aveValue) for item in valueList]  ##适应度
        sumFit = sum(fitnessList)
        pList = [item / sumFit for item in fitnessList]  ##被选概率
        notBeEmployed = []
        for bee in self.onLookList:
            if random.random() < 0.3:  # 百分之30的概率不成为雇佣蜂
                notBeEmployed.append(bee)
            else:
                follow = self.roulette(pList)
                self.st[follow].addFolloers(bee)
        self.onLookList = copy.deepcopy(notBeEmployed)  # 现在依然是雇佣峰的

    def scoutsPro(self):
        newScoutList = []
        for bee in self.scoutsList:
            newSt = Nectar([self.getFeasibleSol(self.bound), self.getFeasibleSol(self.bound)])
            for i in range(30):  ##侦察峰的探索次数
                temPos = [self.getFeasibleSol(self.bound), self.getFeasibleSol(self.bound)]
                temSt = Nectar(temPos)
                if temSt.getValue() > newSt.getValue():
                    newSt = temSt
            if newSt.getValue() < self.bestValue:
                self.st.append(newSt)
                self.bestValue = newSt.getValue()
                self.bestPos = newSt.getPos()
                print("目前最优", self.bestValue, "|", self.bestPos)
            else:  # r
                if random.random() < 0.3:
                    self.onLookList.append(bee)
                else:
                    newScoutList.append(bee)
        self.scoutsList = copy.deepcopy(newScoutList)

    def solve(self):
        self.initial()  # 初始化完成
        ##雇佣蜂行动找下一个峰源
        for i in range(self.I):
            self.employedPro()
            ##跟随峰行动
            self.onLookerPro()
            ##侦察峰出动
            self.scoutsPro()
            self.resultRecord.append(self.bestValue)
            if i in self.pictureIndex:  ##抓取蜜源快照
                self.resultForPicture.append(self.st)

    def showResult(self):
        ##画优化过程曲线
        fig = plt.figure()
        plt.plot(range(len(self.resultRecord)), self.resultRecord)
        plt.title("优化过程曲线")
        ##画快照
        fig1 = plt.figure()
        for i in range(len(self.pictureIndex)):
            ax = fig1.add_subplot(3, 4, i + 1)
            x, y = [], []
            for st in self.resultForPicture[i]:
                x.append(st.getPos()[0])
                y.append(st.getPos()[1])
            ax.scatter(x, y)
            plt.xlabel("第%s次迭代" % (self.pictureIndex[i]))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 初始化蜂群的大小，追随者数目，雇佣者的数目，侦察者的数目
    I = 200  ##最好大于100好画图
    B = 120
    onLookers = int(0.5 * B)
    employedBee = int(0.5 * B)
    scouts = 80
    myPros = testPro()
    grienWankFunc = myPros.GriewankFunction
    bounds = [-10, 10]
    myBCA = ABC(grienWankFunc, B, onLookers, employedBee, scouts, I, bounds)
    myBCA.solve()
    print(myBCA.bestValue)
    print(myBCA.bestPos)
    myBCA.showResult()
    myPros.showGriewankFunction(3, bounds)
    myPros.showGeneralizeRastrigin(3, bounds)
    myPros.showAckleyFunction(3, bounds)
