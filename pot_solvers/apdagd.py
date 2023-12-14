import numpy as np
import numpy.linalg as LA
from scipy.special import xlogy
from .round import round_matrix
import warnings
import time


class F:
    """
    Primal objective function.
    """

    def __init__(self, c, gamma):
        """
        Args:
            c: vectorized version of cost matrix C, shape (n ** 2,)
            gamma: regularization parameter
        """
        self.nsq = c.shape[0]
        self.n = int(np.sqrt(self.nsq))
        self.gamma = gamma
        # Augment c so that it is followed by 2 * n zeros
        self.c = np.hstack((c, np.zeros(2 * self.n)))

    def __call__(self, x):
        """
        Args:
            x: vec(X), p, q, of shape (n ** 2 + 2 * n)
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
            C: Cost matrix, of shape (n, n)
            gamma: regularization parameter, positive scalar
            a: source distribution, of shape (n,)
            b: destination distribution, of shape (n,)
            s: total transported mass
        """
        self.gamma = gamma
        # Gibbs kernel
        self.gibbs = np.exp(-C / gamma)
        self.a = a
        self.b = b
        self.s = s
        self.n = a.shape[0]
        self.nsq = self.n ** 2
        # This is b in the "Ax = b" part of APDAGD
        self.stack = np.hstack((a, b, s))
        self.f = F(C.flatten(), gamma)

    def x_lamb(self, y, z, t):
        """
        x(lambda) in dual formulation. This is the solution to maximizing the
        Lagrangian. Here the dual variables lambda = (y, z).
        Args:
            y: column dual variables, of shape (n,)
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
            x: flattened variables, of shape (n ** 2 + 2 * n)
        """
        X = x[:self.nsq].reshape(self.n, self.n)
        p = x[self.nsq:self.nsq + self.n]
        q = x[self.nsq + self.n:]
        return np.hstack((X.sum(axis=1) + p, X.sum(axis=0) + q, np.sum(X)))

    def __call__(self, lamb):
        """
        args:
            lamb: of shape (2 * n + 1)
        """
        # Get dual variables
        y, z, t = lamb[:self.n], lamb[self.n:2 * self.n], lamb[-1]
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
        y, z, t = lamb[:self.n], lamb[self.n:2 * self.n], lamb[-1]
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z, t)
        return self.stack - self._A_x(x_lamb)


def apdagd(a_dist,
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
           ):
    """
    Adaptive Primal-Dual Accelerated Gradient Descent algorithm.

    args:
        a_dist: Source distribution. Non-neg array of shape (n, ).
        b_dist: Destination distribution. Non-neg array of shape (n, ).
        C: Cost matrix. Non-neg array of shape (n, n)
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

    returns:
        X: Partial optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates

    Reference: Dvurechensky et al., 2018. Algorithms 3 and 4.
    Source: https://github.com/chervud/AGD-vs-Sinkhorn/blob/master/APDAGD.m
    """

    n = a_dist.shape[0]
    assert b_dist.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a_dist) > 0, "APDAGD requires the source a to be positive"
    assert np.min(b_dist) > 0, "APDAGD requires the destination b to be positive"
    assert s <= min(a_dist.sum(), b_dist.sum()), \
        "s must be less than or equal to the sum of a and b"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    # Regularization parameter gamma
    gamma = gamma if gamma is not None else tol / (4 * np.log(n))
    if verbose:
        print("Regularization parameter: gamma = {:.2e}".format(gamma))

    # NEW
    # eps_p = tol / (8 * np.log(n))
    # eps_r = np.inf if a_dist.sum() <= 1 else 8 * (a_dist.sum() - s) / (a_dist.sum() - 1)
    # eps_c = np.inf if b_dist.sum() <= 1 else 8 * (b_dist.sum() - s) / (b_dist.sum() - 1)
    # eps_p = min(eps_p, eps_r, eps_c)

    # Tolerance for the dual problem
    apdagd_tol = tol / (8 * np.max(C))
    if verbose:
        print("Tolerance for duality gap        : {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||X1 + p - a||_2   : {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||X.T 1 + q - b||_2: {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||1.T X 1 - s||_2  : {:.2e}".format(apdagd_tol / 2))

    ##### STEP 1: Rescale the marginals #####
    a_dist_tilde = (1 - apdagd_tol / 8) * \
                   (a_dist + (apdagd_tol / (n * (8 - apdagd_tol))))
    b_dist_tilde = (1 - apdagd_tol / 8) * \
                   (b_dist + (apdagd_tol / (n * (8 - apdagd_tol))))

    ##### STEP 2: Reformulate #####

    # Create primal and dual objectives
    phi = Phi(C, gamma, a_dist_tilde, b_dist_tilde, s=s)
    f = phi.f

    start_time = time.time()
    logs = {
        "L": 1,  # Lipschitz smoothness param
        "lambda": [np.zeros(2 * n + 1)],  # also y
        "zeta": np.zeros(2 * n + 1),  # also z
        "eta": np.zeros(2 * n + 1),  # also x
        "x": [np.zeros(n ** 2 + 2 * n)],  # primal solution
        "phi": [phi(np.zeros(2 * n + 1))],  # dual objective
        "f": [],  # primal objective
        "duality_gap": [],  # duality gap = f + phi
        "alpha": 0,  # alpha_i in Nesterov
        "beta": 0,  # beta_i in Nesterov
        "CX": [],
        "transport_matrix": [],
        "row_cons_err": [],
        "col_cons_err": [],
        "total_mass_err": [],
    }

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
                y, z, t = lamb_k_1[:n], lamb_k_1[n:2 * n], lamb_k_1[-1]
                x_k_1 = tau_k * phi.x_lamb(y, z, t) + (1 - tau_k) * x_k
                if save_iterates:
                    logs["x"].append(x_k_1)
                    nsq = int(n ** 2)
                    X_ = x_k_1[:nsq].reshape(n, n)
                    p_ = x_k_1[nsq:nsq + n]
                    q_ = x_k_1[nsq + n:]
                    logs["row_cons_err"].append(
                        np.linalg.norm(X_.sum(1) + p_ - a_dist_tilde, ord=2)
                    )
                    logs["col_cons_err"].append(
                        np.linalg.norm(X_.sum(0) + q_ - b_dist_tilde, ord=2)
                    )
                    logs["total_mass_err"].append(
                        abs(X_.sum() - s)
                    )
                else:
                    logs["x"] = [x_k_1]

                if save_iterates:
                    X = round_matrix(x_k_1, a_dist, b_dist, s)[0]
                    logs["transport_matrix"].append(X)
                    logs["CX"].append(np.sum(C * X))

                logs["f"].append(f(x_k_1))
                logs["duality_gap"].append(
                    np.abs(logs["f"][-1] + logs["phi"][-1]))

                break

        # Evaluate the stopping conditions
        X_k_1 = x_k_1[: int(n ** 2)].reshape((n, n))
        p_k_1 = x_k_1[int(n ** 2): int(n ** 2) + n]
        q_k_1 = x_k_1[int(n ** 2) + n:]
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
            # print("Iter = {:-5d} | "
            #       "Duality gap = {:.2e} | "
            #       "Row cons err = {:.2e} | "
            #       "Col cons err = {:.2e} | "
            #       "Tot mass err = {:.2e} | "
            #       .format(k, error1, error2, error3, error4))

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
    X, _, _ = round_matrix(x=logs["x"][-1], r=a_dist, c=b_dist, s=s)

    return X, logs
