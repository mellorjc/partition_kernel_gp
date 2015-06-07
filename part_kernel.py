from numpy import ones, zeros, where, argmin
from numpy.random import choice
from numpy.linalg import norm
from scipy.stats import binom, uniform
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean
from scipy.sparse.linalg import LinearOperator, cg
from random import randint

class FastKernel:
    def __init__(self, X, m=500, h=3, distance=None, sigma=0.1, eps=0.001, maxiter=20):
        self.maxiter = maxiter
        self.v = None
        self.m = m
        self.h = h
        self.sigma = sigma
        self._K = None
        self.eps = eps
        if distance is None:
            self.d = euclidean
        else:
            self.d = distance
        self.selected = False
        self.center_list = []
        # the number of centers for each m
        self.num_c = zeros(self.m, dtype=int)
        if len(X.shape) == 1:
            y = 1
        else:
            x, y = X.shape
        if y is None:
            y = 1
        self.dim_masks = zeros((y, self.m))

    def _select_centers(self, X):
        if self.selected:
            return
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        for i in range(self.m):
            if not self.selected:
                # s is the number of centers to have
                s = min(2**randint(1, self.h), max(1, X.shape[0]//2))
                if len(X.shape) == 1:
                    centers = X[choice(X.shape[0], s, False)]
                else:
                    centers = X[choice(X.shape[0], s, False), :]
                self.center_list.append(centers)
                # the dimensions to care about
                if len(X.shape) == 1:
                    y = 1
                else:
                    x, y = X.shape
                if y is None:
                    y = 1
                dim_mask = binom.rvs(1, 0.5, size=y)
                self.dim_masks[:, i] = dim_mask
                self.num_c[i] = s
        self.selected = True

    def K(self, X):
        # the cluster class assigned to each example use
        self._select_centers(X)
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        c = zeros((X.shape[0], self.m))
        for i in range(self.m):
            
            centers = self.center_list[i]
            dim_mask = self.dim_masks[:, i]
            # calculate distances
            X_mask = dim_mask.reshape((1, dim_mask.size))*X
            c_mask = dim_mask.reshape((1, dim_mask.size))*centers
            dists = pairwise_distances(X_mask, c_mask, metric=self.d)
            c[:, i] = argmin(dists, axis=1)
        return c, self.num_c

    def K2(self, X, X2):
        res = zeros((X.shape[0], X2.shape[0]))
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        if len(X2.shape) == 1:
            X2 = X2.reshape((X2.shape[0], 1))
        for k in range(self.m):
            centers = self.center_list[k]
            dim_mask = self.dim_masks[:, k]
            # calculate distances
            X_mask = dim_mask.reshape((1, dim_mask.size))*X
            X2_mask = dim_mask.reshape((1, dim_mask.size))*X2
            c_mask = dim_mask.reshape((1, dim_mask.size))*centers
            # find the distance to the centers for matrix X
            dists = pairwise_distances(X_mask, c_mask, metric=self.d)
            # find the center which is closest for each example in X
            c = argmin(dists, axis=1)
            # find the distance to the centers for matrix X2
            dists = pairwise_distances(X2_mask, c_mask, metric=self.d)
            # find the center which is closest for each example in X2
            c2 = argmin(dists, axis=1)
            for i, c_v in enumerate(c):
                for j, c_v2 in enumerate(c2):
                    if c_v == c_v2:
                        res[i, j] += 1.
        res /= self.m
        return res

    def Ky(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        res = zeros(X.shape[0])
        c, num_c = self.K(X)
        for i in range(self.m):
            for j in range(num_c[i]):
                ind = c[:, i] == j
                res[ind] += y[ind].sum()
        res /= self.m
        return res

    def B(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))
        res = zeros(X.shape[0])
        c, num_c = self.K(X)
        for i in range(self.m):
            for j in range(num_c[i]):
                ind = c[:, i] == j
                res[ind] += (1./(float(where(ind)[0].size) + self.sigma))*y[ind].sum()
        res /= self.m
        res = (1./self.sigma)*y - res
        return res
    
    def solve(self, X, y):
        # see Wikipedia preconditioned conjugate gradient method
        # for details
        r = y - self.Ky(X,x0) - self.sigma*x0
        z = self.B(X, r)
        p = z
        d = z.dot(r)
        while True:
            Ap = self.Ky(X,p) + self.sigma*p
            a = r.dot(z)/(p.dot(Ap))
            x = x + a*p
            r2 = r - a*Ap
            if norm(r2) < self.eps:
                break
            z2 = self.B(X, r)
            d2 = z2.dot(r2)
            b = d2/d
            p = z2 + b*p
            r = r2
            z = z2
            d = d2
        return x

    def predict_mean(self, X2, X, y):
        A = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.Ky(X, x) - self.sigma*x)
        M = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.B(X, x))
        if self.v is None:
            self.v, info = cg(A, y, M=M, maxiter=10)
        k = self.K2(X2, X)
        return k.dot(v)
    
    def predict_var(self, X2, X, y):
        vs = zeros(X2.shape[0])
        for i in range(X2.shape[0]):
            v = self.K2(X2[i, :], X2[i, :])
            A = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.Ky(X, x) - self.sigma*x)
            M = LinearOperator((X.shape[0], X.shape[0]), lambda x: self.B(X, x))
            k_star = self.K2(X2[i, :], X)
            tmp = cg(A, k_star.T, M=M, maxiter=self.maxiter)
            vs[i] = v - k_star.dot(tmp)
        return vs
