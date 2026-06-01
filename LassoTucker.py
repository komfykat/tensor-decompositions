from sklearn.linear_model import MultiTaskLasso
import tensorly as tl
import numpy as np


class LassoTucker:
    def __init__(self, ranks=None, n_iter_max=100, tol=1e-4,
                 random_state=None, alpha=0.01, disjoint_modes=None):
        self.ranks = ranks
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.random_state = random_state
        self.alpha = alpha
        self.disjoint_modes = disjoint_modes if disjoint_modes else []

        self.core = None
        self.factors = None
        self.fit_history_ = []

    def compute_core(self, X):
        factors_pinv = [np.linalg.pinv(tl.to_numpy(f)) for f in self.factors]
        self.core = tl.tensor(tl.tenalg.multi_mode_dot(X, factors_pinv))
        return self

    def _update_factor_lasso(self, X, n: int) -> np.ndarray:
        ndim = tl.ndim(X)

        X_unfold = tl.to_numpy(tl.unfold(X, n))

        G_unfold = tl.to_numpy(tl.unfold(self.core, n))

        kron = None
        for m in range(ndim - 1, -1, -1):
            if m == n:
                continue
            f_m = tl.to_numpy(self.factors[m])
            kron = f_m if kron is None else np.kron(kron, f_m)

        design = (G_unfold @ kron.T).T
        target = X_unfold.T

        if not np.isfinite(design).all() or not np.isfinite(target).all():
            print(f"  Warning: NaN/Inf in mode {n}, skipping update")
            return self.factors[n]

        alpha_scaled = (
            self.alpha
            * np.max(np.abs(design.T @ target))
            / max(design.shape[0], 1)
        )

        model = MultiTaskLasso(
            alpha=alpha_scaled,
            fit_intercept=False,
            max_iter=2000,
            tol=1e-6,
        )
        model.fit(design, target)

        result = model.coef_  

        if not np.any(result):
            print(f"  Warning: Lasso zeroed out factor {n}, falling back to SVD")
            return self._update_factor_standard(X, n)

        return tl.tensor(result)

    def _update_factor_standard(self, X, n: int) -> np.ndarray:
        Y = X
        for m in range(tl.ndim(X)):
            if m != n:
                Y = tl.tenalg.mode_dot(Y, tl.to_numpy(self.factors[m]).T, m)
        M = tl.to_numpy(tl.unfold(Y, n))  
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        return tl.tensor(U[:, :self.ranks[n]])

    def compute_fit(self, X) -> float:
        X_rec = tl.tucker_to_tensor((self.core, self.factors))
        return float(1.0 - tl.norm(X - X_rec) / (tl.norm(X) + 1e-12))

    def fit(self, X) -> "LassoTucker":
        X = tl.tensor(X, dtype=float)
        ndim = tl.ndim(X)

        if self.ranks is None:
            self.ranks = list(X.shape)
        else:
            self.ranks = list(self.ranks)


        _, self.factors = tl.decomposition.tucker(
            X, rank=self.ranks, init="svd", random_state=self.random_state
        )
        self.compute_core(X)

        prev_fit = self.compute_fit(X)
        self.fit_history_.append(prev_fit)

        for _iter in range(self.n_iter_max):
            for n in range(ndim):
                if n in self.disjoint_modes:
                    self.factors[n] = self._update_factor_lasso(X, n)
                else:
                    self.factors[n] = self._update_factor_standard(X, n)
                self.compute_core(X)

            current_fit = self.compute_fit(X)
            self.fit_history_.append(current_fit)

            rel_change = abs(current_fit - prev_fit) / (abs(prev_fit) + 1e-12)
            if rel_change < self.tol:
                break
            prev_fit = current_fit

        return self

    def predict(self):
        return tl.tucker_to_tensor((self.core, self.factors))

    def fit_transform(self, X):
        self.fit(X)
        return self.core, self.factors