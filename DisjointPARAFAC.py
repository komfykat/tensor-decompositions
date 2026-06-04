import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao

class DisjointPARAFAC:
    def __init__(self, rank=5, disjoint_modes=None, 
                 n_iter_max=100, tol=1e-6, random_state=None, 
                 norm_tol=1e-12, rcond=1e-10):
        self.rank = rank
        self.disjoint_modes = set(disjoint_modes) if disjoint_modes else set()
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.norm_tol = norm_tol
        self.rcond = rcond
        
        self.weights = None
        self.A = self.B = self.C = None

    def make_disjoint(self, M):
        """Project factor to disjoint manifold: keep max-abs entry per row."""
        if M.size == 0: return M
        idx = np.argmax(np.abs(M), axis=1)
        M_new = np.zeros_like(M)
        M_new[np.arange(M.shape[0]), idx] = M[np.arange(M.shape[0]), idx]
        return M_new

    def _normalize(self, F, weights):
        norms = np.linalg.norm(F, axis=0)
        norms = np.maximum(norms, self.norm_tol)
        F /= norms
        weights *= norms
        return F, weights

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3: raise ValueError("Only 3D Tensors supported")
        R = self.rank

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # SVD initialization (all modes)
        self.A = np.linalg.svd(tl.unfold(X, 0), full_matrices=False)[0][:, :R]
        self.B = np.linalg.svd(tl.unfold(X, 1), full_matrices=False)[0][:, :R]
        self.C = np.linalg.svd(tl.unfold(X, 2), full_matrices=False)[0][:, :R]
        self.weights = np.ones(R, dtype=np.float64)

        prev_fit = -np.inf
        for it in range(self.n_iter_max):
            # --- Update A ---
            V = (self.C.T @ self.C) * (self.B.T @ self.B) + 1e-12 * np.eye(R)
            K = khatri_rao([self.C, self.B])
            self.A = tl.unfold(X, 0) @ K @ np.linalg.pinv(V, rcond=self.rcond)
            if 0 in self.disjoint_modes:
                self.A = self.make_disjoint(self.A)
            # self.A, self.weights = self._normalize(self.A, self.weights)

            # --- Update B ---
            V = (self.C.T @ self.C) * (self.A.T @ self.A) + 1e-12 * np.eye(R)
            K = khatri_rao([self.C, self.A])
            self.B = tl.unfold(X, 1) @ K @ np.linalg.pinv(V, rcond=self.rcond)
            if 1 in self.disjoint_modes:
                self.B = self.make_disjoint(self.B)
            # self.B, self.weights = self._normalize(self.B, self.weights)

            # --- Update C ---
            V = (self.B.T @ self.B) * (self.A.T @ self.A) + 1e-12 * np.eye(R)
            K = khatri_rao([self.B, self.A])
            self.C = tl.unfold(X, 2) @ K @ np.linalg.pinv(V, rcond=self.rcond)
            if 2 in self.disjoint_modes:
                self.C = self.make_disjoint(self.C)
            # self.C, self.weights = self._normalize(self.C, self.weights)

            # --- Convergence ---
            X_rec = tl.cp_to_tensor((self.weights, [self.A, self.B, self.C]))
            err = np.linalg.norm(X - X_rec)
            current_fit = 1.0 - err / (np.linalg.norm(X) + 1e-12)
            
            if not np.isfinite(current_fit):
                current_fit = prev_fit if it > 0 else 0.0
                
            if it > 0 and abs(current_fit - prev_fit) < self.tol:
                break
            prev_fit = current_fit

        return self

    def compute_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_rec = tl.cp_to_tensor((self.weights, [self.A, self.B, self.C]))
        err = np.linalg.norm(X - X_rec)
        fit = 1.0 - err / (np.linalg.norm(X) + 1e-12)
        return fit if np.isfinite(fit) else 0.0

    def fit_transform(self, X):
        self.fit(X)
        return self.weights, [self.A, self.B, self.C]