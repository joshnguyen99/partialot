import numpy as np
import numpy.linalg as LA
from .round import round_matrix
from scipy.special import xlogy
import warnings

FLOAT_TYPE = np.double


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
        self.c = c
        self.gamma = gamma

    def __call__(self, x):
        return np.dot(self.c, x) + self.gamma * xlogy(x, x).sum()

    def grad(self, x):
        return self.c + self.gamma + np.log(x) + self.gamma

    def hess(self, x):
        return self.gamma * np.diag(1 / x)


class Phi:
    """
    Dual objective function.
    """

    def __init__(self, C, gamma, a, b):
        """
        args:
            C: Cost matrix, of shape (n, n)
            gamma: regularization parameter, positive scalar
            a: source distribution, of shape (n,)
            b: destination distribution, of shape (n,)
        """
        self.gamma = gamma
        # Gibbs kernel
        self.gibbs = np.exp(-C / gamma)
        self.a = a
        self.b = b
        self.n = a.shape[0]
        # This is b in the "Ax = b" part of APDAGD
        self.stack = np.hstack((a, b))
        self.f = F(C.flatten(), gamma)

    def x_lamb(self, y, z):
        """
        x(lambda) in dual formulation. This is the solution to maximizing the
        Lagrangian. Here the dual variables lambda = (y, z).
        Args:
            y: column dual variables, of shape (n,)
            z: row dual variables, of shape (n,)
        """
        X = (1 / np.e) * np.exp(-z / self.gamma) * (np.exp(-y / self.gamma) * self.gibbs.T).T
        return X.flatten()

    def _A_transpose_lambda(self, y, z):
        """
        Implicitly compute A^T lambda.
        args:
            y: dual variables corresponding to column constraints, of shape (n,)
            z: dual variables corresponding to row constraints, of shape (n,)
        """
        return np.add.outer(y, z).flatten()

    def _A_x(self, x):
        """
        Implicitly compute A x, equal to [X 1, X^T 1]
        args:
            x: flattened variables, of shape (n^2,)
        """
        X = x.reshape(self.n, self.n)
        return np.hstack((X.sum(axis=1), X.sum(axis=0)))

    def __call__(self, lamb):
        # Get dual variables
        y, z = lamb[:self.n], lamb[self.n:]
        val = np.dot(self.a, y) + np.dot(self.b, z)
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z).flatten()
        val -= self.f(x_lamb)
        val -= np.dot(self._A_transpose_lambda(y, z), x_lamb)
        return val

    def grad(self, lamb):
        """
        Gradient of the dual objective function with respect to lambda.
        args:
            lamb: dual variables, of shape (2n,)
        """
        y, z = lamb[:self.n], lamb[self.n:]
        # Maximizing Lagrangian to get x(lambda)
        x_lamb = self.x_lamb(y, z)
        return self.stack - self._A_x(x_lamb)


def apdagd(a_dist,
           b_dist,
           C,
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
        a_dist: Source distribution. Non-neg array of shape (n, ), sums to 1.
        b_dist: Destination distribution. Non-neg array of shape (n, ), sums to 1.
        C: Cost matrix. Non-neg array of shape (n, n).
        num_iters: Maximum number of Sinkhorn iterations. Int.
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

    This implementation is based on Lin, Ho and Jordan 2019.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates

    Reference: Dvurechensky et al., 2018. Algorithms 3 and 4.
    Source: https://github.com/chervud/AGD-vs-Sinkhorn/blob/master/APDAGD.m
    """

    n = a_dist.shape[0]
    assert b_dist.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a_dist) > 0, "Sinkhorn requires the source a to be positive"
    assert np.min(
        b_dist) > 0, "Sinkhorn requires the destination b to be positive"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    a = a_dist.astype(FLOAT_TYPE)
    b = b_dist.astype(FLOAT_TYPE)
    C = C.astype(FLOAT_TYPE)

    # Regularization parameter gamma
    gamma = gamma if gamma is not None else tol / (4 * np.log(n))
    if verbose:
        print("Regularization parameter: gamma = {}".format(gamma))

    # Tolerance for the dual problem
    apdagd_tol = tol / (8 * np.max(C))
    if verbose:
        print("Tolerance for duality gap : {:.2e}".format(apdagd_tol / 2))
        print("Tolerance for ||Ax - b||_2: {:.2e}".format(apdagd_tol / 2))

    ##### STEP 1: Rescale the marginals #####
    a_dist_tilde = (1 - apdagd_tol / 8) * \
                   (a_dist + (apdagd_tol / (n * (8 - apdagd_tol))))
    b_dist_tilde = (1 - apdagd_tol / 8) * \
                   (b_dist + (apdagd_tol / (n * (8 - apdagd_tol))))

    ##### STEP 2: Reformulate #####

    # Create primal and dual objectives
    phi = Phi(C, gamma, a_dist_tilde, b_dist_tilde)
    f = phi.f

    logs = {
        "L": 1,  # Lipschitz smoothness param
        "lambda": [np.zeros(2 * n, dtype=FLOAT_TYPE)],  # also y
        "zeta": np.zeros(2 * n, dtype=FLOAT_TYPE),  # also z
        "eta": np.zeros(2 * n, dtype=FLOAT_TYPE),  # also x
        "x": [np.zeros(n ** 2, dtype=FLOAT_TYPE)],  # primal solution
        "phi": [phi(np.zeros(2 * n, dtype=FLOAT_TYPE))],  # dual objective
        "f": [],  # primal objective
        "duality_gap": [],  # duality gap = f + phi
        "alpha": 0,  # alpha_i in Nesterov
        "beta": 0,  # beta_i in Nesterov
        "CX": [],
        "transport_matrix": []
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
                y, z = lamb_k_1[:n], lamb_k_1[n:]
                x_k_1 = tau_k * phi.x_lamb(y, z) + (1 - tau_k) * x_k
                if save_iterates:
                    logs["x"].append(x_k_1)
                else:
                    logs["x"] = [x_k_1]

                if save_iterates:
                    X = round_matrix(x_k_1.reshape(n, n), a_dist, b_dist)
                    logs["transport_matrix"].append(X)
                    logs["CX"].append(np.sum(C * X))

                logs["f"].append(f(x_k_1))
                logs["duality_gap"].append(
                    np.abs(logs["f"][-1] + logs["phi"][-1]))

                break

        # Evaluate the stopping conditions
        X_k_1 = x_k_1.reshape((n, n))
        # Error 1: marginal constraints
        error1 = np.sum((X_k_1.sum(1) - a_dist_tilde) ** 2)
        error1 += np.sum((X_k_1.sum(0) - b_dist_tilde) ** 2)
        error1 = np.sqrt(error1)
        # Error 2: dual optimality
        error2 = logs["duality_gap"][-1]

        # Print status
        if verbose and k % print_every == 0:
            print("Iter = {:-5d} | "
                  "Duality gap = {:.2e} | "
                  "||Ax - b|| = {:.2e} | "
                  "L estimate = {:-6.1f}"
                  .format(k, error2, error1, M_k))

        # Terminate if tolerance is achieved
        converged = error1 <= apdagd_tol / 2 and error2 <= apdagd_tol / 2
        if check_termination and converged:
            if verbose:
                print("APDAGD converged after {} iterations".format(k))
            break
        if k >= num_iters and not converged:
            warnings.warn("WARNING: APDAGD did not converge after {} iterations"
                          .format(num_iters))

    ##### STEP 4: Round the primal solution #####
    X = round_matrix(X=logs["x"][-1].reshape(n, n), a=a_dist, b=b_dist)

    return X, logs
