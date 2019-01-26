# -*- coding:utf-8 -*-
import numpy as np
from hmm import HMM

A = np.array([[ 0.7,  0.3],
              [ 0.4,  0.6]])
B = np.array([[ 0.5, 0.4, 0.1],
              [ 0.1, 0.3, 0.6]])
hmm = HMM(A, B, [0.3,0.7])
# print(hmm.generate_data(5, seed=2018))
observations, states = hmm.generate_data(T=10, seed=2019)
print('observations: {}'.format(observations))
print('hidden states: {}'.format(states))
#  概率计算问题
print('backward prob: {}'.format(hmm.backward(observations)[1]))
print('forward prob: {}'.format(hmm.forward(observations)[1]))

# 学习问题
model = HMM(np.array([[0.5, 0.5],
                          [0.5, 0.5]]),
                np.array([[0.4, 0.4, 0.2],
                          [0.2, 0.3, 0.5]]),
                np.array([0.5, 0.5])
                )
a, b, pi, count = model.baum_welch(observations, threshold=0.1)
print('EM iteration: {}'.format(count))
print('a: {}'.format(a))
print('b: {}'.format(b))
print('pi: {}'.format(pi))

# 预测问题
print("predict: {}".format(hmm.viterbi(observations)))
