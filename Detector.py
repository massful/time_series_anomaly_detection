#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import json
from collections import deque
import bisect
import random

import numpy as np

from Distance import Distance

class DimError(Exception):
    pass

def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print "@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__)
        back = func(*args, **args2)
        print "@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__)
        print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back
    return newFunc

class TimeSeriesAnormalyDetector(object):
    '''时间序列异常检验模型
        1. 本算法基于以下几个思路：
            1.1 欧氏距离和余弦距离衡量向量间相似度
            1.2 指数衰减的方式计算当前时间点与之前T-1个时间点的相似度
            1.3 加入周期因素，周期相似度融入3点动态弯曲
            1.4 平滑异常与周期异常加权平均
            1.5 加入上期异常修正因子，使异常后数值迅速归位
        2. 参数说明
            alpha: 异常修正因子权重
            beta: 平滑异常与周期异常权重
            gamma: 指数平滑权重
            w_cos: 余弦距离占总距离的权重
            w_ecld: 欧氏距离占总距离的权重
            period: 序列周期，可不指定
            dim: 多元时间序列的维度
            var_weight: 多元变量各维权重；仅用于欧氏距离计算
        3. 使用方式：
            3.1 fit和fit_detect方法: 将已有数据写入对象，初始化对象属性，以后可能还会有自动参数优化等；fit_detect返回训练数据的异常检验结果
            3.2 detect方法接收一条新的数据，并判定异常值
            3.3 anormal_rank方法返回当前点的异常度在全局的排名
    
    '''
    def __init__(self, alpha, beta, gamma, w_cos, w_ecld, dim, period=None, var_weight = None, sa_keep=3, var_up_down = None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_cos = w_cos
        self.w_ecld = w_ecld
        self.period = period
        self.dim = dim
        self.var_weight = var_weight
        self.sa_keep = sa_keep
        self.var_up_down = var_up_down
        self.init()

    def init(self):
        self.q_series = [] #时间窗
        self.last_sa = 0
        self._std = [0.00001] * self.dim
        if not self.period:
            self.period = 60 #默认保留60个点
            self.is_period = False
        else:
            self.is_period = True
        if not self.var_weight:
            self.var_weight = [1] * self.dim
        else:
            self._check_weight()
        if not self.var_up_down:
            self.var_up_down = [1] * self.dim
        else:
            self._check_var_up_down()
        #用优先列表做历史异常值记录
        self.anormal_heap = []

    def init2(self, dataset):
        self._std = np.std(np.array(dataset), axis=0) #init std
        print 'init std: %s'%(self._std)

    def _check_weight(self):
        # check if the sum of var_weight not equal to 1; if 1, zoom it 
        if abs(sum(self.var_weight) - 1) <= 0.00001:
            self.var_weight = [self.dim * self.var_weight[i] for i in range(self.dim)]
        else:
            raise DimError

    def _check_size(self, dataset):
        if not [1 for row in dataset if len(row) != self.dim]:
            pass
        else:
            raise DimError

    def _check_var_up_down(self):
        # set var up down
        try:
            self.var_up_down = [1 if i in self.var_up_down else 0 for i in range(self.dim)]
        except:
            raise DimError

    def fit(self, dataset, is_normalize = False):
        #用已有数据初始化对象
        #TODO : 自动优化参数
        self._check_size(dataset)
        self.init2(dataset)
        for row in dataset:
            self.detect(row, is_mask = True)

    def fit_detect(self, dataset, is_normalize = False):
        self._check_size(dataset)
        self.init2(dataset)
        for row in dataset:
            yield self.detect(row )

    def diff(self, x1, x2):
        # 计算标准化欧氏距离与余弦距离
        ecld_dis = Distance.ecld_distance(x1, x2, self._std, self.var_weight)        
        cos_dis = Distance.cos_distance(x1, x2 )
        #print 'ecld dis: %s, cos dis : %s'%(ecld_dis, cos_dis)
        return self.w_cos * cos_dis + self.w_ecld * ecld_dis

    def curr_diff(self, x):
        # 指数平滑
        return sum([self.gamma * (1 - self.gamma)**i * self.diff(x, y) for i,y in enumerate(self.q_series[:0:-1])])

    #def modify_sa(self , sigma=0.5):
    #    return sum([sigma * (1 - sigma) ** i * y  for i,y in enumerate(self.last_sa[::-1])])

    def update(self, x, sa, is_mask):
        # 更新对象属性
        self.last_sa = sa 
        if len(self.q_series) < self.period + 1:
            self.q_series.append(x)
        else:
            self.q_series = self.q_series[1:] + self.q_series[:1]
            self.q_series[-1] = x
        self._std = np.std(self.q_series[1:], axis = 0)
        if not is_mask:
            bisect.insort_left(self.anormal_heap, self.last_sa)

    def _up_down(self, diff):
        return filter(lambda x:x!=0 , [w * (1 if d>0 else -1) for w,d in zip(self.var_up_down, diff)])

    def last_up_down(self, x):
    # 上一个点的值, 环比
        diff = [x[i] - self.q_series[-1][i] for i in xrange(self.dim)]
        return self._up_down(diff)

    def period_up_down(self, x):
    # 同期值对比
        diff = [x[i] - self.q_series[1][i] for i in xrange(self.dim)]
        return self._up_down(diff)

    def detect(self, x, show_up_down = True, is_mask=False):
        # 执行一次异常检验，并更新对象
        direction_last, direction_period = None, None
        if len(self.q_series) == 0:
            sa = 0
        elif (len(self.q_series) < self.period) or (not self.is_period):
            curr_sa = self.curr_diff(x)
            modify_sa = (1+self.alpha) * ( curr_sa ) - self.alpha * self.last_sa
            sa = min(curr_sa, modify_sa)
            direction_last = self.last_up_down(x)
        else:
            curr_sa = self.beta * self.curr_diff(x) + (1 - self.beta) * min(self.diff(x, self.q_series[0]), self.diff(x, self.q_series[1]), self.diff(x, self.q_series[2]))
            modify_sa = (1+self.alpha) * curr_sa - self.alpha * self.last_sa
            sa = min(curr_sa, modify_sa)
            direction_last = self.last_up_down(x)
            direction_period = self.period_up_down(x)

        sa = max(sa, 0.0)
        self.update(x, sa, is_mask)
        return sa, direction_last, direction_period if show_up_down else sa

    def anormal_rank(self):
        #判断当前异常值在历史上是TopN
        total = len(self.anormal_heap)
        return total - self.anormal_heap.index(self.last_sa), total

    @property
    def last_point(self):
        return self.q_series[-1] if len(self.q_series) >= 1 else None

    @property
    def last_period_point(self):
        return self.q_series[1] if (self.is_period and len(self.q_series) == self.period+1) else None

    @property
    def std(self):
        return self._std
        
random.seed(1)
def test():
    test_seed = [[random.uniform(0,1000), random.uniform(0,10), random.uniform(0,1), random.uniform(0,1), random.randint(0,500)] for i in range(20)]    
    detector = TimeSeriesAnormalyDetector(0.2, 0.5, 0.6, 0.5, 0.5, 5, 4 )
    result = detector.fit_detect(test_seed)
    for row in [[random.uniform(0,1000), random.uniform(0,10), random.uniform(0,1), random.uniform(0,1), random.randint(0,500)] for i in range(20)]:
        print 'current row: %s'%row
        print 'anormal score: %s, point_by_point_direction : %s, period_by_period_direction : %s'%detector.detect(row)
        print 'anormal rank : %s, %s'%detector.anormal_rank()
        print 'point_by_point : %s \n period_by_period : %s'%(detector.last_point, detector.last_period_point)


if __name__ == '__main__':
    test()
