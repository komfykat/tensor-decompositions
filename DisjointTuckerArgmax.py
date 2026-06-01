import numpy as np
import tensorly as tl

class DisjointTuckerArgmax:
    def __init__(self, ranks=None, n_iter_max = 100, tol=1e-4, disjoint_modes=None, n_runs=30):
        self.ranks = ranks
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.disjoint_modes = disjoint_modes if disjoint_modes is not None else []
        self.n_runs = n_runs
        
        self.core = None
        self.factors = None

    def make_disjoint(self, M: tl.tensor):
        M_new = np.zeros_like(M)
        idx = np.argmax(np.abs(M), axis=1)

        for i in range(M.shape[0]):
            M_new[i, idx[i]] = M[i, idx[i]]
            
        norms = np.linalg.norm(M_new, axis=0, keepdims=True) + 1e-12
        M_new /= norms

        return M_new

    def compute_core(self, X):
        factors_pinv = [np.linalg.pinv(tl.to_numpy(f)) for f in self.factors]
        self.core = tl.tensor(tl.tenalg.multi_mode_dot(X, factors_pinv))
        return self
    
    def predict(self):
        return tl.tenalg.multi_mode_dot(self.core, self.factors)

    def fit(self, X: tl.tensor):
        best_fit = None
        best_decomposition = None
        prev_fit = 0
        for run in range(self.n_runs):
            _, self.factors = tl.decomposition.tucker(X, rank=self.ranks, random_state=run)
            for iter in range(self.n_iter_max):
                for n in range(len(self.factors)):
                    Y = X
                    # Y = tl.tenalg.multi_mode_dot(X, [self.factors[i] for i in range(len(self.factors)) if i != n], transpose=True)
                    for i in range(len(self.factors)): 
                        if i != n:
                            Y = tl.tenalg.mode_dot(Y, self.factors[i].T, i)
                    self.factors[n], _, _ = np.linalg.svd(tl.unfold(Y, n), full_matrices=True)
                    if self.ranks != None:
                        self.factors[n] = self.factors[n][:, :self.ranks[n]]
                    if n in self.disjoint_modes:
                        self.factors[n] = self.make_disjoint(self.factors[n])
            self.compute_core(X)
            fit = self.compute_fit(X)
            if best_fit is None or fit > best_fit:
                best_fit = fit
                best_decomposition = (self.core.copy(), [f.copy() for f in self.factors])
            if np.abs(fit - prev_fit) / (np.abs(prev_fit) + 1e-12) < self.tol:
                break
            prev_fit = fit
        self.core, self.factors = best_decomposition
        return self
    
    def compute_fit(self, X: tl.tensor):
        return 1 - tl.norm(self.predict() - X)/tl.norm(X)

    def fit_transform(self, X: tl.tensor):
        self.fit(X)
        return self.core, self.factors