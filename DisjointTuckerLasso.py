from sklearn.linear_model import Lasso
import tensorly as tl
import numpy as np

class DisjointTuckerLasso:
    def __init__(self, ranks=None, n_iter_max=100, tol=1e-4, random_state=None, alpha=0.01):
        self.ranks = ranks
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha

        self.core = None
        self.U = None
        self.V = None
        self.W = None

    def compute_core(self, X: tl.tensor):
        self.core = tl.tenalg.multi_mode_dot(X, [self.U, self.V, self.W], transpose=True)
        return self

    def _enforce_disjoint(self, factor: tl.tensor):
        idx = tl.argmax(tl.abs(factor), axis=1)
        new_factor = tl.zeros_like(factor)
        rows = tl.arange(factor.shape[0])
        new_factor[rows, idx] = factor[rows, idx]
        return new_factor


    def fit(self, X: tl.tensor):
        X = tl.tensor(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.ranks is None:
            self.ranks = X.shape
            
        self.core, [self.U, self.V, self.W] = tl.decomposition.tucker(
           X, rank=self.ranks, init='svd', random_state=self.random_state
        )


        for _ in range(self.n_iter_max):
            X_mat = tl.kron(self.W, self.V) @ tl.unfold(self.core, 0).T
            Y_mat = tl.unfold(X, 0).T
            alpha = self.alpha * np.max(np.abs(X_mat.T @ Y_mat)) / X_mat.shape[0]
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, random_state=self.random_state)
            model.fit(tl.to_numpy(X_mat), tl.to_numpy(Y_mat))
            self.U = model.coef_

            X_mat = tl.kron(self.W, self.U) @ tl.unfold(self.core, 1).T
            Y_mat = tl.unfold(X, 1).T
            alpha = self.alpha * np.max(np.abs(X_mat.T @ Y_mat)) / X_mat.shape[0]
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, random_state=self.random_state)
            model.fit(tl.to_numpy(X_mat), tl.to_numpy(Y_mat))
            self.V = model.coef_

            X_mat = tl.kron(self.V, self.U) @ tl.unfold(self.core, 2).T
            Y_mat = tl.unfold(X, 2).T
            alpha = self.alpha * np.max(np.abs(X_mat.T @ Y_mat)) / X_mat.shape[0]
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000, random_state=self.random_state)
            model.fit(tl.to_numpy(X_mat), tl.to_numpy(Y_mat))
            self.W = model.coef_
            
            self.compute_core(X)
            
                
        return self

    def fit_transform(self, X: tl.tensor):
        self.fit(X)
        return self.core, [self.U, self.V, self.W]