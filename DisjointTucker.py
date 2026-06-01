import numpy as np
import tensorly as tl
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

class DisjointTucker:
    def __init__(self, ranks=None, n_iter_max=100, tol=1e-4, 
                 random_state=None, disjoint_modes=None):
        self.ranks = ranks
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.disjoint_modes = disjoint_modes
        
        self.core = None
        self.factors = None
        self.fit_history_ = []

    def _disjoint_pca(self, X: np.ndarray, n_components: int, n_iter_max=100, tol=1e-6):
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        X = X - X.mean(axis=0)

        partition = rng.randint(0, n_components, size=p)
        for k in range(n_components):
            if not np.any(partition == k):
                partition[rng.randint(0, p)] = k

        loadings = np.zeros((p, n_components))
        prev_obj = -np.inf

        for iteration in range(n_iter_max):
            explained_var = np.zeros(n_components)
            for k in range(n_components):
                idx = np.where(partition == k)[0]
                if len(idx) == 0: continue
                
                R_k = (X[:, idx].T @ X[:, idx]) / (n - 1)
                eigvals, eigvecs = np.linalg.eigh(R_k)
                idx_max = np.argmax(eigvals)
                explained_var[k] = eigvals[idx_max]
                v_k = eigvecs[:, idx_max]
                loadings[idx, k] = v_k * np.sqrt(explained_var[k])

            obj = explained_var.sum()
            if iteration > 0 and abs(obj - prev_obj) < tol:
                break
            prev_obj = obj

            L_potential = np.zeros((p, n_components))
            for k in range(n_components):
                idx = np.where(partition == k)[0]
                if len(idx) == 0: continue
                v_k = loadings[idx, k] / (np.sqrt(explained_var[k]) + 1e-12)
                u_k = X[:, idx] @ v_k
                L_potential[:, k] = (X.T @ u_k) / ((n - 1) * np.sqrt(explained_var[k] + 1e-12))

            new_partition = np.argmax(L_potential**2, axis=1)
            for k in range(n_components):
                if not np.any(new_partition == k):
                    new_partition[np.argmax(L_potential[:, k])] = k

            if np.array_equal(partition, new_partition):
                partition = new_partition
                break
            partition = new_partition

        final_loadings = np.zeros((p, n_components))
        for k in range(n_components):
            idx = np.where(partition == k)[0]
            if len(idx) > 0:
                final_loadings[idx, k] = loadings[idx, k]
                
        return final_loadings

    def _update_factor_disjoint(self, X, n):
        R_n = self.ranks[n]
        
        Y = X
        for m in range(X.ndim):
            if m != n:
                Y = tl.tenalg.mode_dot(Y, self.factors[m].T, m)
                
        M = tl.unfold(Y, n)
        X_pca = tl.to_numpy(M.T) 
        
        B = self._disjoint_pca(X_pca, n_components=R_n, 
                               tol=self.tol, n_iter_max=self.n_iter_max)
        
        norms = np.linalg.norm(B, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        B = B / norms
        
        return tl.tensor(B)
    
    def _update_factor_standard(self, X, n):
        R_n = self.ranks[n]
        Y = X
        for m in range(X.ndim):
            if m != n:
                Y = tl.tenalg.mode_dot(Y, self.factors[m].T, m)
        B, _, _ = np.linalg.svd(tl.unfold(Y, n), full_matrices=True)
        B = B[:, :R_n]
        return B



    def compute_core(self, X):
        factors_pinv = [np.linalg.pinv(tl.to_numpy(f)) for f in self.factors]
        self.core = tl.tensor(tl.tenalg.multi_mode_dot(X, factors_pinv))
        return self
    
    def compute_fit(self, X):
        X_rec = tl.tucker_to_tensor((self.core, self.factors))
        return 1 - tl.norm(X - X_rec) / (tl.norm(X) + 1e-12)

    def fit(self, X):
        X = tl.tensor(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.disjoint_modes is None:
            self.disjoint_modes = list(range(X.ndim))
        
        if self.ranks is None:
            self.ranks = list(X.shape)
        else:
            self.ranks = list(self.ranks)
        
        self.core, self.factors = tl.decomposition.tucker(
            X, rank=self.ranks, init='svd', random_state=self.random_state
        ) 

        prev_fit = self.compute_fit(X)
        for iter in range(self.n_iter_max):
            for n in range(X.ndim):
                if n in self.disjoint_modes:
                    self.factors[n] = self._update_factor_disjoint(X, n)
                else:
                    self.factors[n] = self._update_factor_standard(X, n)
            
            self.compute_core(X)
            
            current_fit = self.compute_fit(X)
            self.fit_history_.append(current_fit)
            
            if abs(current_fit - prev_fit) / (abs(prev_fit) + 1e-12) < self.tol:
                break
            prev_fit = current_fit
            
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.core, self.factors
    

# Код сгенерирован с помощью нейросети Qwen