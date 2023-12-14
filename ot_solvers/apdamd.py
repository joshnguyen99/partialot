import numpy as np
import numpy.linalg as LA
from scipy.special import xlogy
from .round import round_matrix
import warnings


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
        # This is b in the "Ax = b" part of APDAMD
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


def apdamd(a_dist,
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
    Adaptive Primal-Dual Accelerated Mirror Descent algorithm.

    args:
        a_dist: Source distribution. Non-neg array of shape (n, ), sums to 1.
        b_dist: Destination distribution. Non-neg array of shape (n, ), sums to 1.
        C: Cost matrix. Non-neg array of shape (n, n)
        num_iters: Maximum number of Sinkhorn iterations. Int.
        verbose: Whether to print progress. Bool.
        print_every: Print progress every this many iterations. Int.
        tol: Primal optimality tolerance: output X s.t. f(X) <= f(X*) + tol.
             Float. This is epsilon in the algorithm.
        gamma: (Optional) Parameter for entropic regularization. If None, then
               gamma is set to tol / (4 * log(n)).
        check_termination: Whether to check termination condition. If True,
                           optimization is terminated when the APDAMD tolerance
                           is reached. If False, APDAMD is run for num_iters.
        save_iterates: Whether to save primal solutions for all iterations. If
                       False, only the final solution is recorded. Can be memory
                       intensive for large n and large num_iters.

    This implementation is based on Lin, Ho and Jordan 2019. It uses delta = n
    and B_phi(lambda1, lambda2) = (1 / 2n) ||lambda1 - lambda2||^2.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates
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

    # Regularization parameter gamma
    gamma = gamma if gamma is not None else tol / (4 * np.log(n))
    if verbose:
        print("Regularization parameter  : gamma = {:.2e}".format(gamma))

    # Tolerance for APDAMD
    apdamd_tol = tol / (8 * np.max(C))
    if verbose:
        print("Tolerance for ||Ax - b||_1: tol   = {:.2e}".format(apdamd_tol / 2))

    ##### STEP 1: Rescale the marginals #####
    a_dist_tilde = (1 - apdamd_tol / 8) * \
                   (a_dist + (apdamd_tol / (n * (8 - apdamd_tol))))
    b_dist_tilde = (1 - apdamd_tol / 8) * \
                   (b_dist + (apdamd_tol / (n * (8 - apdamd_tol))))

    ##### STEP 2: Reformulate #####

    # Create primal and dual objectives
    phi = Phi(C, gamma, a_dist_tilde, b_dist_tilde)
    f = phi.f

    logs = {
        "L": 1,  # Lipschitz smoothness param
        "lambda": [np.zeros(2 * n)],  # in mirror descent
        "z": np.zeros(2 * n),  # in mirror descent
        "mu": np.zeros(2 * n),  # in mirror descent
        "x": [np.zeros(n ** 2)],  # primal solution
        "phi": [phi(np.zeros(2 * n))],  # dual objective
        "f": [],  # primal objective
        "duality_gap": [],  # duality gap = f + phi
        "alpha": 0,  # in acceleration
        "alpha_bar": 0,  # in acceleration
    }

    ##### STEP 3: Run APDAGD until (apdamd_tol / 2) accuracy #####
    delta = n
    for t in range(1, num_iters + 1):

        M_t = logs["L"] / 2
        alpha_bar_t = logs["alpha_bar"]
        z_t = logs["z"]
        lamb_t = logs["lambda"][-1]
        x_t = logs["x"][-1]

        # Line search
        while True:
            # Initial guess for L
            M_t = 2 * M_t

            # Compute the step size
            alpha_t_1 = (1 + np.sqrt(1 + 4 * delta * M_t * alpha_bar_t)) / (2 * delta * M_t)

            # Compute the average coefficient
            alpha_bar_t_1 = alpha_bar_t + alpha_t_1

            # Compute the first average step
            mu_t_1 = (alpha_t_1 * z_t + alpha_bar_t * lamb_t) / alpha_bar_t_1

            # Compute the mirror descent
            grad_phi_mu_t_1 = phi.grad(mu_t_1)
            z_t_1 = z_t - alpha_t_1 * grad_phi_mu_t_1

            # Compute the second average step
            lamb_t_1 = (alpha_t_1 * z_t_1 + alpha_bar_t * lamb_t) / alpha_bar_t_1

            # Evaluate smoothness
            lhs = phi(lamb_t_1)
            rhs = phi(mu_t_1) + np.dot(grad_phi_mu_t_1, lamb_t_1 - mu_t_1) + \
                0.5 * M_t * LA.norm(lamb_t_1 - mu_t_1, ord=np.inf) ** 2
            if lhs <= rhs:
                # Line search condition fulfilled
                if save_iterates:
                    logs["lambda"].append(lamb_t_1)
                else:
                    logs["lambda"] = [lamb_t_1]
                logs["mu"] = mu_t_1
                logs["z"] = z_t_1
                logs["phi"].append(lhs)
                logs["L"] = M_t / 2
                logs["alpha"] = alpha_t_1
                logs["alpha_bar"] = alpha_bar_t_1

                # Recover primal solution
                y, z = lamb_t_1[:n], lamb_t_1[n:]
                x_t_1 = (alpha_t_1 * phi.x_lamb(y, z) + alpha_bar_t * x_t) / alpha_bar_t_1
                if save_iterates:
                    logs["x"].append(x_t_1)
                else:
                    logs["x"] = [x_t_1]
                logs["f"].append(f(x_t_1))
                logs["duality_gap"].append(np.abs(logs["f"][-1] + logs["phi"][-1]))

                break

        # Evaluate the stopping condition
        X_t_1 = x_t_1.reshape((n, n))
        # Error 1: marginal constraints
        error1 = LA.norm(X_t_1.sum(1) - a_dist_tilde, ord=1)
        error1 += LA.norm(X_t_1.sum(0) - b_dist_tilde, ord=1)

        # Print status
        if verbose and t % print_every == 0:
            print("Iter = {:-5d} | "
                  "Duality gap = {:.2e} | "
                  "||Ax - b||_1 = {:.2e} | "
                  "L estimate = {:-6.1f}"
                  .format(t, logs["duality_gap"][-1], error1, M_t))

        # Terminate if tolerance is achieved
        converged = error1 <= apdamd_tol / 2
        if check_termination and converged:
            if verbose:
                print("APDAMD converged after {} iterations".format(t))
            break
        if t >= num_iters and not converged:
            warnings.warn("WARNING: APDAMD did not converge after {} iterations"
                          .format(num_iters))
    ##### STEP 4: Round the primal solution #####
    X = round_matrix(X=logs["x"][-1].reshape(n, n), a=a_dist, b=b_dist)

    return X, logs
