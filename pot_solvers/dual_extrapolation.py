import numpy as np
import scipy as sp
from numpy import linalg as LA
from .round import round_matrix
from .apdagd import apdagd

# matplotlib inline
np.random.seed(1)


class Dual_Extrapolation_POT:
    def __init__(self, n, d, b, alpha=1 / 9, beta=1 / 18, e_coeff=5, eps=1e-5,
                 T=None, save_iterates=False, verbose=True, print_every=100):
        self.n = n
        self.nsq = n ** 2
        self.b = b
        self.alpha = alpha  # theoretically : 1 / kappa
        self.beta = beta  # theoretically : 1 / (2 * kappa)
        self.e_coeff = e_coeff  # theoretically : 5
        self.d = d
        self.r = b[:self.n].copy()
        self.c = b[self.n:2 * self.n].copy()
        self.s = b[-1:].copy()
        self.dnorm = LA.norm(d, np.inf)
        self.eps = eps
        self.Theta = 60 * self.dnorm * np.log(n) + 6 * self.dnorm
        self.save_iterates = save_iterates
        self.verbose = verbose
        self.print_every = print_every

        self.T = T if T is not None else np.ceil(36 * self.Theta / self.eps).astype(int)

    # Calculate Ax !!! Have to change according to our problem
    def multA(self, x):
        """
        Implicitly compute A x, equal to [X 1, X^T 1]
        args:
            x: flattened variables, of shape (n ** 2 + 2 * n) containing X, p and q
        """
        X = x[:self.nsq].reshape(self.n, self.n)
        p = x[self.nsq:self.nsq + self.n]
        q = x[self.nsq + self.n:]
        return np.hstack((X.sum(axis=1) + p, X.sum(axis=0) + q, np.sum(X)))

    def multAt(self, lamb):
        """
        Implicitly compute A^T lambda.
        args:
            lamb: dual variables, of shape (2 * n + 1), containing y, z and t
        """
        y, z, t = lamb[:self.n], lamb[self.n:2 * self.n], lamb[-1]

        return np.hstack((np.add.outer(y, z).flatten() + t, y, z))

    # Calculate g_x
    def g_x(self, x, y):
        dnorm = self.dnorm
        result = self.d + 23 * dnorm * self.multAt(y)
        return result

    # Calculate g_y
    def g_y(self, x, y):
        b = self.b
        dnorm = self.dnorm
        result = -23 * dnorm * (self.multA(x) - b)
        return result

    # AM step
    # Solve minimization problem of type: min H(x,y) = <v,x> + <u, y> + r(x,y)
    def alternating_minimization(self, v, u, iters=None):
        """
        AM procedure for dual extrapolation.
        args:
            v: of shape n ** 2 + 2 * n
            u: of shape 2 * n + 1
            iters: number of AM iterations. If None, then it is calculated
                   based on the theoretical value.
        """
        n = self.n
        dnorm = self.dnorm
        e_coeff = self.e_coeff

        M = 24 * np.log(
            (840 * self.dnorm / (self.eps ** 2) + 6 / self.eps) * self.Theta +
            1336 * self.dnorm / 9
        )
        iters = iters if iters is not None else np.ceil(M).astype(int)

        # Initialization
        x = 1 / (n ** 2 + 2 * n) * np.ones(n ** 2 + 2 * n)
        y = np.zeros(2 * n + 1)

        out_x = x.copy()
        out_y = y.copy()

        for t in range(iters):
            xi = 1 / (2 * e_coeff) * (1 / (2 * dnorm) * v + self.multAt(y ** 2))
            x = sp.special.softmax(-xi)
            y = np.minimum(np.ones(2 * self.n + 1),
                           np.maximum(-np.ones(2 * self.n + 1), -u / (4 * dnorm * self.multA(x))))
            out_x = x
            out_y = y

        return out_x, out_y

    def solve(self):
        n = self.n
        dnorm = self.dnorm
        alpha = self.alpha
        beta = self.beta
        e_coeff = self.e_coeff

        # Initialization
        # For now, use the result of APDAGD as a warm start
        X, _ = apdagd(a_dist=self.r, b_dist=self.c, s=self.s,
                      C=self.d[:n ** 2].reshape(n, n),
                      num_iters=1000, save_iterates=False,
                      gamma=1e-2)
        x = np.hstack((X.flatten(), np.zeros(2 * n)))

        s_x = np.zeros(n ** 2 + 2 * n) + 1e-6
        s_y = np.zeros(2 * n + 1) + 1e-6

        den = np.sum(self.r) + np.sum(self.c) - self.s
        out_x = x / den
        out_y = s_y.copy()
        self.b /= den

        logs = {
            "x": [out_x] if self.save_iterates else [],
            "y": [out_y] if self.save_iterates else [],
            "transport_matrix": [round_matrix(out_x, self.r, self.c, self.s)[0]] if
            self.save_iterates else [],
        }

        # Calculate \nabla_x r and \nabla_y r
        r_grad_x = 4 * e_coeff * dnorm * (1 - np.log(n ** 2 + 2 * n)) * np.ones(2 * n + n ** 2)
        r_grad_y = np.zeros(2 * n + 1)

        for k in range(1, self.T + 1):
            v = s_x - r_grad_x
            u = s_y - r_grad_y

            z_x, z_y = self.alternating_minimization(v, u)
            v += alpha * self.g_x(z_x, z_y)
            u += alpha * self.g_y(z_x, z_y)

            w_x, w_y = self.alternating_minimization(v, u)
            s_x += beta * self.g_x(w_x, w_y)
            s_y += beta * self.g_y(w_x, w_y)

            # Averaging
            out_x = k / (k + 1) * out_x + 1 / (k + 1) * w_x
            out_y = k / (k + 1) * out_y + 1 / (k + 1) * w_y

            if self.save_iterates:
                logs["x"].append(out_x * den)
                logs["y"].append(out_y)
                logs["transport_matrix"].append(
                    round_matrix(out_x * den, self.r, self.c, self.s)
                )

            if self.verbose and k % self.print_every == 0:
                # TODO: Log errors
                print("Iter = {:-5d}".format(k))

        return round_matrix(out_x * den, self.r, self.c, self.s)[0], logs


def dual_extrapolation(a_dist,
                       b_dist,
                       C,
                       s,
                       num_iters=10000,
                       verbose=False,
                       print_every=1000,
                       tol=1e-5,
                       save_iterates=True,
                       ):
    n = a_dist.shape[0]
    assert b_dist.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a_dist) > 0, "DE requires the source a to be positive"
    assert np.min(b_dist) > 0, "DE requires the destination b to be positive"
    assert s <= min(a_dist.sum(), b_dist.sum()), \
        "s must be less than or equal to the sum of a and b"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    d = np.hstack((C.flatten(), np.zeros(2 * n)))
    b = np.hstack((a_dist, b_dist, s))

    solver = Dual_Extrapolation_POT(n,
                                    d,
                                    b,
                                    eps=tol,
                                    T=num_iters,
                                    save_iterates=save_iterates,
                                    verbose=verbose,
                                    print_every=print_every)

    return solver.solve()
