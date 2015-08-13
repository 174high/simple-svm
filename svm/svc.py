# -*- coding: UTF-8 -*-
__author__ = 'Draonfly'
"""Support Vector Classifier"""
import numpy as np

class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1,
                 probability=False, tol=1e-3, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.SV_index = None
        self.SVs = None
        self.label_SVs = None
        self.b = 0

    def _kernel_trans(self, X, x_i):
        """实现kernel的转换"""
        m, n = X.shape()
        K = np.mat(np.zeros((m, 1)))
        if self.kernel == 'linear':
            K = X * x_i.T
        elif self.kernel == 'rbf':
            for j in range(m):
                delta_row = X[j, :] - x_i
                K[j] = delta_row * delta_row.T
                # exp(-gamma||x-x'||^2)
            K = np.exp(-self.gamma * K)
        else:
            raise NameError('%s is not recognized' % self.kernel)
        return K

    def _calcEk(self, optstruct, k):
        fXk = float(np.multiply(optstruct.alphas, optstruct.y).T * optstruct.K[:, k] + optstruct.b)
        Ek = fXk - float(optstruct.y[k])
        return Ek

    def _selectJ(self, i, optstruct, Ei):
        j = -1
        max_deltaE = 0
        Ej = 0
        # 由于e_cache初始化为mat(zeros(m, 2)) [1, Ei]用于标明Ei是否放入了e_cache中
        optstruct.e_cache[i] = [1, Ei]
        valid_e_cache_list = np.nonzero(optstruct.e_cache[:, 0].A)[0]
        if len(valid_e_cache_list) > 1:
            for k in valid_e_cache_list:
                if k == i:
                    continue
                Ek = self._calcEk(optstruct, k)
                deltaE = abs(Ei-Ek)
                if deltaE > max_deltaE:
                    j, Ej, max_deltaE = k, Ek, deltaE
            return j, Ej
        else:
            j = self._selectJrand(i, optstruct.m)
            Ej = self._calcEk(optstruct, j)
        return j, Ej

    def _selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def _clip_alpha(self, alpha_j, H, L):
        """对alpha_j_new_unc进行剪辑，使其满足L<=alpha_j_new<=H"""
        if alpha_j > H:
            alpha_j = H
        if alpha_j < L:
            alpha_j = L
        return alpha_j

    def _updateEk(self, optstruct, k):
        Ek = self._calcEk(optstruct, k)
        optstruct.e_cachep[k] = Ek

    def _inner_loop(self, i, optstruct):
        Ei = self._calcEk(optstruct, i)
        # 判断alpha_i是否违反KKT条件
        if ((optstruct.y[i] * Ei < -optstruct.tol) and (optstruct.alphas[i] < optstruct.C)) or \
                ((optstruct.y[i]* Ei > optstruct.tol) and (optstruct.alphas[i] > 0)):
            j, Ej = self._selectJ(i, optstruct, Ei)
            # 复制一份 防止alphas[i]的修改覆盖alpha_i_old
            alpha_i_old = optstruct.alphas[i].copy()
            alpha_j_old = optstruct.alphas[j].copy()
            # alpha_j_new 需要满足不等式约束，即 L <= alpah_j_new <= H
            if optstruct.y[i] != optstruct.y[j]:
                L = max(0, optstruct.alphas[j] - optstruct.alphas[i])
                H = min(optstruct.C, optstruct.C + optstruct.alphas[j] - optstruct.alphas[i])
            else:
                L = max(0, optstruct.alphas[j] + optstruct.alphas[i] - optstruct.C)
                H = min(optstruct.C, optstruct.alphas[j] + optstruct.alphas[i])
            if L == H:
                # 由于L==H，alpha_j不能进一步被优化
                print('L==H')
                return 0
            eta = 2.0 * optstruct.K[i, j] - optstruct.K[i, i] - optstruct.K[j, j]
            if eta >= 0:
                # alpha_2_new_unc = alpha_2_old - y2(E1-E2)/eta
                # eta = 2K12 - K11 - K22 = -||Z(x1) - Z(x2)||^2 <= 0
                # 又因为eta为分母，所以eta需要 < 0
                print('eta>=0')
                return 0
            optstruct.alphas[j] -= optstruct.y[j] * (Ei - Ej) / eta
            optstruct.alphas[j] = self._clip_alpha(optstruct.alphas[j], H, L)
            self._updateEk(optstruct, j)
            if abs(optstruct.alphas[j] - alpha_j_old) < 0.00001:
                # 如果alpha_j变化很小，那么alpha_i也就基本不变，在这里跳出可以提高程序运行速度
                print('j not moving enough')
                return 0
            optstruct.alphas[i] += optstruct.y[j]*optstruct.y[i]*(alpha_j_old-optstruct.alphas[j])
            self._updateEk(optstruct, i)
            b1 = optstruct.b - Ei - optstruct.y[i]*(optstruct.alphas[i]-alpha_i_old)*optstruct.K[i, i] -\
                optstruct.y[j]*(optstruct.alphas[j]-alpha_j_old)*optstruct.K[i, j]
            b2 = optstruct.b - Ej - optstruct.y[j]*(optstruct.alphas[j]-alpha_j_old)*optstruct.K[j, j] -\
                optstruct.y[i]*(optstruct.alphas[i]-alpha_i_old)*optstruct.K[i, j]
            # 如果alpha1 alpha2对应的点是free支持向量，那么b1 == b2
            # 如果alpha1 alpha2是0或者C，那么b1 b2以及他们之间的值都是符合KKT条件的，这时可以选择他们的中点
            if 0 < optstruct.alphas[i] < optstruct.C:
                optstruct.b = b1
            elif 0 < optstruct.alphas[j] < optstruct.C:
                optstruct.b = b2
            else:
                optstruct.b = (b1 + b2)/2.0
            return 1
        else:
            # 如果alpha_i满足KKT条件
            return 0

    def _smo(self, X, y):
        optstruct = _OptStruct(np.mat(X), np.mat(y).transpose(), self.C, self.tol)
        iters = 0
        entire_set = True
        alpha_pairs_changed = 0
        while (iters < self.max_iter) and (alpha_pairs_changed > 0 or entire_set):
            alpha_pairs_changed = 0
            # 遍历所有值
            if entire_set:
                for i in range(optstruct.m):
                    alpha_pairs_changed += self._inner_loop(i, optstruct)
                print('fullset, iter: %d i:%d, pairs changed %d' % (iters, i, alpha_pairs_changed))
                iters += 1
            # 遍历满足条件 0<alpha<C 的点
            else:
                non_bound_alphas = np.nonzero((optstruct.alphas.A > 0) * (optstruct.alphas.A <optstruct.C))[0]
                for i in non_bound_alphas:
                    alpha_pairs_changed += self._inner_loop(i, optstruct)
                    print('non-bound, iter: %d i:%d, pairs changed %d' % (iters, i, alpha_pairs_changed))
                iters += 1
            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print('iteration number: %d' % iters)
        SV_index = np.nonzero(optstruct.alphas.A >0)[0]
        SVs = optstruct.X[SV_index]
        label_SV = optstruct.y[SV_index]
        print('there arre %d Support Vectors' % SVs.shape[0])
        return optstruct.b, optstruct.alphas, SVs, label_SV, SV_index

    def fit(self, X, y):
        self.b, self.alphas, self.SVs, self.label_SVs, self.SV_index = self._smo(X, y)

    def get_w(self):
        pass

    def predict(self, X):
        X = np.mat(X)
        m, n = X.shape
        result = []
        for i in range(m):
            kernel_val = self._kernel_trans(self.SVs, X[i, :])
            predict = kernel_val.T * np.multiply(self.label_SVs, self.alphas[self.SV_index]) + self.b
            result.append(1 if predict >= 0 else -1)
        return result

    def predict_prob(self, X):
        # todo 用LR来计算概率
        pass


class _OptStruct:
    """
    中间结构，便于处理数据
    """
    def __init__(self, X, y, C, tol, kernel_trans):
        """
        X: data matrix [sample counts, feature counts]
        y: label matrix [sample counts, 1]
        """
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.m = X.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros(self.m, 2))
        self.K = np.mat((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :])

