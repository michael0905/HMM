# -*- coding:utf-8 -*-
import numpy as np

class HMM():
    """
    一阶隐马尔可夫模型
    A：ndarray,状态转移概率矩阵
    B：ndarray,观测概率矩阵
    pi: ndarray,初始状态概率向量
    """
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def generate_data(self, T, seed=0):
        """
        生成观测序列和隐藏序列，长度为T
        """
        np.random.seed(seed)
        n = len(self.pi) # 隐状态个数
        m = len(self.B[0]) # 观测状态个数
        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = np.random.choice(n, 1, p=self.pi)[0]
        for t in range(1, T):
            states[t] = np.random.choice(n, 1, p=self.A[states[t-1]])[0]
        for i, v in enumerate(states):
            observations[i] = np.random.choice(m, 1, p=self.B[v])[0]
        return observations, states

    def forward(self, observations):
        """
        前向算法，返回序列概率
        observations: 1darray
        """
        T = len(observations) # 序列长度
        N = len(self.pi) # 隐状态个数
        p = np.zeros([N,T])
        p[:,0] = self.pi * self.B[:,observations[0]] # 点乘
        for t in range(1,T):
            for n in range(N):
                # 前向概率：t时刻隐状态转移到n，然后输出o的概率
                p[n,t] =  np.dot(p[:,t-1], self.A[:,n]) * self.B[n,observations[t]]
        return p, np.sum(p[:,-1])

    def backward(self, observations):
        """
        后向算法，返回序列概率
        observations: 1darray
        """
        T = len(observations) # 序列长度
        N = len(self.pi) # 隐状态个数
        p = np.zeros([N,T])
        p[:,-1] = [1] * N # T+1步后向概率为1
        for t in reversed(range(T-1)):
            for n in range(N):
                # 后向概率：t时刻状态为n，t+1到T时刻观测序列为O(t+1)...O(T)的概率
                p[n,t] = np.sum(p[:,t+1] * self.A[n,:] * self.B[:, observations[t+1]])
        return p, np.sum(p[:,0] * self.pi * self.B[:,observations[0]])

    def baum_welch(self, observations, threshold=0.05):
        """
        Baum-Welch算法求解学习问题：给定观测序列，学习模型参数(A,B,pi)
        observations: 1darray
        """

        T = len(observations) # 序列长度
        N = len(self.pi) # 隐状态个数
        O = self.B.shape[1] # 观测状态个数
        count = 0
        while True:
            count += 1
            forward, _ = self.forward(observations) # N*T
            backward, _ = self.backward(observations) # N*T

            xi = np.zeros((N,N,T-1))
            for t in range(T-1):
                denominator = np.dot(np.dot(forward[:, t].T, self.A) * self.B[:, observations[t+1]].T, backward[:, t+1])
                for i in range(N):
                    nominator = forward[i,t] * self.A[i,:] * self.B[:, observations[t+1]].T * backward[:, t+1].T
                    xi[i,:,t] = nominator / denominator

            gamma = np.zeros((N,T))
            for t in range(T):
                denominator = np.dot(forward[:,t].T, backward[:,t])
                for i in range(N):
                    nominator = forward[i,t] * backward[i,t]
                    gamma[i,t] = nominator / denominator

            next_pi = gamma[:,0]
            next_A = np.sum(xi, axis=2) / np.sum(gamma[:,:-1], axis=1)
            next_B = np.zeros((N,O))
            for o in range(O):
                mask = (observations == o)
                next_B[:,o] = np.sum(gamma[:,mask], axis=1) / np.sum(gamma, axis=1)

            if np.max(np.abs(self.A - next_A)) < threshold and \
                np.max(np.abs(self.B - next_B)) < threshold and np.max(np.abs(self.pi - next_pi)) < threshold:
                break

            self.A = next_A
            self.B = next_B
            self.pi = next_pi

        return next_A, next_B, next_pi, count


    def viterbi(self, observations):
        """
        维特比算法求解预测问题：给定观测序列observations，求最有可能的状态序列。
        返回状态序列
        """
        T = len(observations) # 序列长度
        N = len(self.pi) # 隐状态个数
        pre_states = np.zeros((N, T-1), dtype=int)
        states = np.zeros((T), dtype=int)
        dp = np.zeros((N, T))
        dp[:, 0] = self.B[:, observations[0]] * self.pi
        for t in range(1, T):
            for j in range(N):
                # t时刻输出o隐状态为j的所有路径的概率,从t-1时刻的任意状态转移到t时刻状态j
                prob = dp[:,t-1] * self.A[:, j] * self.B[j,observations[t]]
                # t时刻输出o隐状态为j的所有路径的概率最大值
                dp[j, t] = np.max(prob)
                # t时刻输出o隐状态为j的最大概率路径，在t-1时刻的状态
                pre_states[j, t-1] = np.argmax(prob)
        states[T-1] = np.argmax(dp[:,-1])
        for i in reversed(range(T-1)):
            states[i] = pre_states[states[i+1],i]
        return states

    
