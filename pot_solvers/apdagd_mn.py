import numpy as np
import numpy.linalg as LA
from scipy.special import xlogy
import warnings
import time


def round_matrix(x, r, c, s):
    def extract(x, m, n):
        X = x[:m * n].reshape(m, n)
        p = x[m * n:m * n + m]
        q = x[m * n + m:]
        return X, p, q

    def enforcing_procedure(r, s, pp, ppp):
        alpha = (np.sum(r) - s) / np.sum(pp)
        if alpha < 1:
            pbar = ppp
        else:
            i = -1
            while np.sum(ppp) <= np.sum(r) - s:
                i += 1
                ppp[i] = r[i]
            ppp[i] = ppp[i] - (np.sum(ppp) - np.sum(r) + s)
            pbar = ppp
        return pbar

    m = np.shape(r)[0]
    n = np.shape(c)[0]
    X, p, q = extract(x, m, n)
    assert np.alltrue(X >= 0)
    assert np.alltrue(p >= 0)
    assert np.alltrue(q >= 0)
    one_m = np.ones(m)
    one_n = np.ones(n)

    pp = np.minimum(p, r)
    qp = np.minimum(q, c)
    alpha = min(1., (np.sum(r) - s) / np.sum(pp))
    beta = min(1., (np.sum(c) - s) / np.sum(qp))

    ppp = alpha * pp
    qpp = beta * qp

    pbar = enforcing_procedure(r, s, pp, ppp)
    qbar = enforcing_procedure(c, s, qp, qpp)

    g = np.minimum(one_m, (r - pbar) / X.sum(1))
    h = np.minimum(one_n, (c - qbar) / X.sum(0))

    Xp = np.dot(np.dot(np.diag(g), X), np.diag(h))

    e1 = (r - pbar) - Xp.sum(1)
    e2 = (c - qbar) - Xp.sum(0)

    Xbar = Xp + (1. / (np.sum(e1))) * np.outer(e1, e2)

    return Xbar, pbar, qbar


class F:
    """
    Primal objective function.
    """

    def __init__(self, c, gamma):
        """
        Args:
            c: vectorized version of cost matrix C, shape (m, n)
            gamma: regularization parameter
        """
        self.m, self.n = c.shape
        c = c.flatten()
        self.gamma = gamma
        # Augment c so that it is followed by (m + n) zeros
        self.c = np.hstack((c, np.zeros(self.m), np.zeros(self.n)))

    def __call__(self, x):
        """
        Args:
            x: vec(X), p, q, of shape (m * n + m + n,)
        """
        return np.dot(self.c, x) + self.gamma * xlogy(x, x).sum()

    def grad(self, x):
        return self.c + self.gamma + np.log(x) + self.gamma

    def hess(self, x):
        return self.gamma * np.diag(1 / x)


class Phi:
    """
    Dual objective function.
    """

    def __init__(self, C, gamma, a, b, s):
        """
        args:
            C: Cost matrix, of shape (m, n)
            gamma: regularization parameter, positive scalar
            a: source distribution, of shape (m,)
            b: destination distribution, of shape (n,)
            s: total transported mass
        """
        self.gamma = gamma
        # Gibbs kernel
        self.gibbs = np.exp(-C / gamma)
        self.a = a
        self.b = b
        self.s = s
        self.m = a.shape[0]
        self.n = b.shape[0]
        self.nsq = self.n ** 2
        self.mn = self.m * self.n
        # This is b in the "Ax = b" part of APDAGD
        self.stack = np.hstack((a, b, s))
        self.f = F(C, gamma)

    def x_lamb(self, y, z, t):
        """
        x(lambda) in dual formulation. This is the solution to maximizing the
        Lagrangian. Here the dual variables lambda = (y, z).
        Args:
            y: column dual variables, of shape (m,)
            z: row dual variables, of shape (n,)
            t: total mass dual variable, scalar
        """
        u, v, w = np.exp(-y / self.gamma), np.exp(-z / self.gamma), np.exp(-t / self.gamma)
        X = (1 / np.e) * w * v * (u * self.gibbs.T).T
        p = (1 / np.e) * u
        q = (1 / np.e) * v
        return np.hstack((X.flatten(), p, q))

    def _A_transpose_lambda(self, y, z, t):
        """
        Implicitly compute A^T lambda.
        args:
            y: dual variables corresponding to column constraints, of shape (n,)
            z: dual variables corresponding to row constraints, of shape (n,)
            t: dual variable corresponding to total mass constraint, scalar
        """
        return np.hstack((np.add.outer(y, z).flatten() + t, y, z))

    def _A_x(self, x):
        """
        Implicitly compute A x, equal to [X 1, X^T 1]
        args:
            x: flattened variables, of shape (m * n + m + n)
        """
        X = x[:self.m * self.n].reshape(self.m, self.n)
        p = x[self.m * self.n:self.m * self.n + self.m]
        q = x[self.m * self.n + self.m:]
        return np.hstack((X.sum(axis=1) + p, X.sum(axis=0) + q, np.sum(X)))

    def __call__(self, lamb):
        """
        args:
            lamb: of shape (m + n + 1)
        """
        # Get dual variables
        y, z, t = lamb[:self.m], lamb[self.m:self.m + self.n], lamb[-1]
        val = np.dot(self.a, y) + np.dot(self.b, z) + t * self.s
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z, t).flatten()
        val -= self.f(x_lamb)
        val -= np.dot(self._A_transpose_lambda(y, z, t), x_lamb)
        return val

    def grad(self, lamb):
        """
        Gradient of the dual objective function with respect to lambda.
        args:
            lamb: dual variables, of shape (2n,)
        """
        y, z, t = lamb[:self.m], lamb[self.m:self.m + self.n], lamb[-1]
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z, t)
        return self.stack - self._A_x(x_lamb)


def apdagd_mn(a_dist,
              b_dist,
              C,
              s,
              num_iters=10000,
              verbose=False,
              print_every=1000,
              tol=1e-5,
              gamma=None,
              check_termination=True,
              save_iterates=True,
              round=True,
              f_star=None,
              ):
    """
    Adaptive Primal-Dual Accelerated Gradient Descent algorithm.

    args:
        a_dist: Source distribution. Non-neg array of shape (m, ).
        b_dist: Destination distribution. Non-neg array of shape (n, ).
        C: Cost matrix. Non-neg array of shape (m, n)
        s: total mass to transport, scalar. Must be less than or equal to the
           sum(a_dist) and sum(b_dist).
        num_iters: Maximum number of APDAGD iterations. Int.
        verbose: Whether to print progress. Bool.
        print_every: Print progress every this many iterations. Int.
        tol: Primal optimality tolerance: output X s.t. f(X) <= f(X*) + tol.
             Float. This is epsilon in the algorithm.
        gamma: (Optional) Parameter for entropic regularization. If None, then
               gamma is set to tol / (4 * log(n)).
        check_termination: Whether to check termination condition. If True,
                           optimization is terminated when the APDAGD tolerance
                           is reached. If False, APDAGD is run for num_iters.
        save_iterates: Whether to save primal solutions for all iterations. If
                       False, only the final solution is recorded. Can be memory
                       intensive for large n and large num_iters.
        round: Whether to applying rounding to the final POT matrix.

    returns:
        X: Partial optimal transport matrix. Non-neg array of shape (m, n).
        logs: dictionary of iterates

    Reference: Dvurechensky et al., 2018. Algorithms 3 and 4.
    Source: https://github.com/chervud/AGD-vs-Sinkhorn/blob/master/APDAGD.m
    """

    m = a_dist.shape[0]
    n = b_dist.shape[0]
    # assert np.min(a_dist) > 0, "APDAGD requires the source a to be positive"
    # assert np.min(b_dist) > 0, "APDAGD requires the destination b to be positive"
    assert s <= min(a_dist.sum(), b_dist.sum()), \
        "s must be less than or equal to the sum of a and b"
    assert C.shape == (m, n), "C must be of shape ({}, {})".format(m, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    # Regularization parameter gamma
    gamma = gamma if gamma is not None else tol / (4 * np.log(max(m, n)))
    if verbose:
        print("Regularization parameter: gamma = {:.2e}".format(gamma))

    # Tolerance for the dual problem
    apdagd_tol = tol / (8 * np.max(C))
    if verbose:
        print("Tolerance for duality gap        : {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||X1 + p - a||_2   : {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||X.T 1 + q - b||_2: {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||1.T X 1 - s||_2  : {:.2e}".format(apdagd_tol / 2))

    ##### STEP 1: Rescale the marginals #####
    a_dist_tilde = (1 - apdagd_tol / 8) * \
                   (a_dist + (apdagd_tol / (m * (8 - apdagd_tol))))
    b_dist_tilde = (1 - apdagd_tol / 8) * \
                   (b_dist + (apdagd_tol / (n * (8 - apdagd_tol))))

    ##### STEP 2: Reformulate #####

    # Create primal and dual objectives
    phi = Phi(C, gamma, a_dist_tilde, b_dist_tilde, s=s)
    f = phi.f

    logs = {
        "L": 1,  # Lipschitz smoothness param
        "lambda": [np.zeros(m + n + 1)],  # also y
        "zeta": np.zeros(m + n + 1),  # also z
        "eta": np.zeros(m + n + 1),  # also x
        "x": [np.zeros(m * n + m + n)],  # primal solution
        "phi": [phi(np.zeros(m + n + 1))],  # dual objective
        "f": [],  # primal objective
        "duality_gap": [],  # duality gap = f + phi
        "alpha": 0,  # alpha_i in Nesterov
        "beta": 0,  # beta_i in Nesterov
        "CX": [],
        "transport_matrix": [],
        "row_cons_err": [],
        "col_cons_err": [],
        "total_mass_err": [],
        "time_per_iter": [],
    }

    start_time = old_time = new_time = time.time()

    ##### STEP 3: Run APDAGD until (apdagd_tol / 2) accuracy #####
    for k in range(1, num_iters + 1):

        ##### STEP 1: Apply APDAGD #####
        beta_k = logs["beta"]
        zeta_k = logs["zeta"]
        eta_k = logs["eta"]
        M_k = logs["L"] / 2
        x_k = logs["x"][-1]

        # Line search
        while True:

            old_time = time.time()

            # Initial guess for L
            M_k = 2 * M_k

            # Solve for the new alpha_{k+1}, beta_{k+1} -> tau_{k+1}
            alpha_k_1 = (1 + np.sqrt(1 + 4 * M_k * beta_k)) / (2 * M_k)
            beta_k_1 = beta_k + alpha_k_1
            tau_k = alpha_k_1 / beta_k_1

            # lambda_{k+1}
            lamb_k_1 = tau_k * zeta_k + (1 - tau_k) * eta_k

            # zeta_{k+1}
            grad_phi_lamb_k_1 = phi.grad(lamb_k_1)
            zeta_k_1 = zeta_k - alpha_k_1 * grad_phi_lamb_k_1

            # eta_{k+1}
            eta_k_1 = tau_k * zeta_k_1 + (1 - tau_k) * eta_k

            new_time = time.time()
            logs["time_per_iter"].append(new_time - old_time)

            # Evaluate smoothness
            lhs = phi(eta_k_1)
            rhs = phi(lamb_k_1) + np.dot(grad_phi_lamb_k_1, eta_k_1 - lamb_k_1) + \
                0.5 * M_k * LA.norm(eta_k_1 - lamb_k_1, ord=2) ** 2

            if lhs <= rhs:
                # Line search condition fulfilled
                if save_iterates:
                    logs["lambda"].append(lamb_k_1)
                else:
                    logs["lambda"] = [lamb_k_1]
                logs["zeta"] = zeta_k_1
                logs["eta"] = eta_k_1
                logs["phi"].append(lhs)
                logs["L"] = M_k / 2
                logs["alpha"] = alpha_k_1
                logs["beta"] = beta_k_1

                # Recover primal solution. This is x_hat_k_1
                y, z, t = lamb_k_1[:m], lamb_k_1[m:m + n], lamb_k_1[-1]
                x_k_1 = tau_k * phi.x_lamb(y, z, t) + (1 - tau_k) * x_k

                if save_iterates:
                    old_time = new_time
                    logs["x"].append(x_k_1)
                    X_ = x_k_1[:m * n].reshape(m, n)
                    p_ = x_k_1[m * n:m * n + m]
                    q_ = x_k_1[m * n + m:]
                    logs["row_cons_err"].append(
                        np.linalg.norm(X_.sum(1) + p_ - a_dist_tilde, ord=2)
                    )
                    logs["col_cons_err"].append(
                        np.linalg.norm(X_.sum(0) + q_ - b_dist_tilde, ord=2)
                    )
                    logs["total_mass_err"].append(
                        abs(X_.sum() - s)
                    )

                    if round:
                        X = round_matrix(x_k_1, a_dist, b_dist, s)[0]
                        logs["transport_matrix"].append(X)
                        CX = np.sum(C * X)
                        logs["CX"].append(CX)

                else:
                    logs["x"] = [x_k_1]

                logs["f"].append(f(x_k_1))
                logs["duality_gap"].append(
                    np.abs(logs["f"][-1] + logs["phi"][-1]))

                if f_star is not None and check_termination is True and CX - f_star < tol:
                    break
                break

        # Evaluate the stopping conditions
        X_k_1 = x_k_1[: int(m * n)].reshape((m, n))
        p_k_1 = x_k_1[int(m * n): int(m * n) + m]
        q_k_1 = x_k_1[int(m * n) + m:]
        # Error 1: Dual optimality
        error1 = logs["duality_gap"][-1]
        # Error 2: Column sum constraint
        error2 = LA.norm(X_k_1.sum(1) + p_k_1 - a_dist_tilde)
        # Error 3: Row sum constraint
        error3 = LA.norm(X_k_1.sum(0) + q_k_1 - b_dist_tilde)
        # Error 4: Total mass constraint
        error4 = LA.norm(np.sum(X_k_1) - s)
        Axmb_err = np.sqrt(error2 ** 2 + error3 ** 2 + error4 ** 2)

        # Print status
        if verbose and k % print_every == 0:
            print("Iter = {:-5d} | "
                  "Duality gap = {:.2e} | "
                  "||Ax - b|| = {:.2e} | "
                  "L estimate = {:-6.1f}"
                  .format(k, error1, Axmb_err, M_k))

        # Terminate if tolerance is achieved
        converged = max(error1, Axmb_err) <= apdagd_tol / 2
        if check_termination and converged:
            if verbose:
                print("APDAGD converged after {} iterations".format(k))
            break
        if k >= num_iters and not converged:
            warnings.warn("WARNING: APDAGD did not converge after {} iterations"
                          .format(num_iters))

    ##### STEP 4: Round the primal solution #####
    running_time = time.time() - start_time
    logs["time"] = running_time
    if round:
        X, _, _ = round_matrix(x=logs["x"][-1], r=a_dist, c=b_dist, s=s)
    else:
        X = logs["x"][-1][:m * n].reshape(m, n)
    return X, logs
