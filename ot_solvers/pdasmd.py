import numpy as np
import numpy.linalg as LA
from scipy.special import xlogy
from .round import round_matrix
import warnings
from scipy.special import softmax


def fullgrad(v, gamma, C, a_dist, b_dist, n):
    """
    Gradient of the semi-dual objective.

    Args:
        v: first dual variable (e.g., lambda), of shape (n,)
        gamma: regularization parameter
        C: cost matrix, of shape (n, n)
        a_dist: source distribution, of shape (n,)
        b_dist: destination distribution, of shape (n,)
        n: dimensions of histograms
    """
    full = -n * (a_dist * (b_dist.T - softmax((v.T - C - gamma) / gamma, axis=1)))
    return full


def batchgrad(v, gamma, C, a_dist, b_dist, idxes, n):
    """
    Batch gradients of the semi-dual objective.

    Args:
        v: first dual variable (e.g., lambda), of shape (n,)
        gamma: regularization parameter
        C: cost matrix, of shape (n, n)
        a_dist: source distribution, of shape (n,)
        b_dist: destination distribution, of shape (n,)
        idxes: indices of the batch
        n: dimensions of histograms
    """
    batch = -n * (a_dist[idxes] * (b_dist.T - softmax((v.T - C[idxes] - gamma) / gamma, axis=1)))
    return batch


def primal_semidual(semi, a_dist, b_dist, C, n, gamma):
    semi = np.reshape(semi, (n, -1))
    p = (softmax((semi.T - C - gamma) / gamma, axis=1) * a_dist).T.reshape(-1, 1)
    return p


def pdasmd(a_dist,
           b_dist,
           C,
           num_iters=10000,
           verbose=False,
           print_every=1000,
           tol=1e-5,
           gamma=None,
           check_termination=True,
           save_iterates=True,
           mirror_descent=True,
           batch_size=1,
           seed=100,
           inner_iters=None):
    """
    Primal-Dual Accelerated Stochastic Proximal Mirror Descent

    Args:
        a_dist: Source distribution. Non-neg array of shape (n, ).
        b_dist: Destination distribution. Non-neg array of shape (n, ).
        C: Cost matrix. Non-neg array of shape (n, n)
        s: total mass to transport, scalar. Must be less than or equal to the
           sum(a_dist) and sum(b_dist).
        num_iters: Maximum number of PDASMD iterations. Int.
        verbose: Whether to print progress. Bool.
        print_every: Print progress every this many iterations. Int.
        tol: Primal optimality tolerance: output X s.t. f(X) <= f(X*) + tol.
             Float. This is epsilon in the algorithm.
        gamma: (Optional) Parameter for entropic regularization. If None, then
               gamma is set to tol / (4 * log(n)).
        check_termination: Whether to check termination condition. If True,
                           optimization is terminated when the PDASMD tolerance
                           is reached. If False, PDASMD is run for num_iters.
        save_iterates: Whether to save primal solutions for all iterations. If
                       False, only the final solution is recorded. Can be memory
                       intensive for large n and large num_iters.
        mirror_descent: Whether to use mirror descent. True corresponds to l_inf norm, False
                        corresponds to l_2 norm.
        batch_size: Batch size.
        seed: Random seed.
        inner_iters: Number of inner iterations.
    """

    n = a_dist.shape[0]
    assert b_dist.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a_dist) > 0, "PDASMD requires the source a to be positive"
    assert np.min(b_dist) > 0, "PDASMD requires the destination b to be positive"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    gamma = gamma if gamma is not None else tol / (4 * np.log(n))
    tol = tol if tol is not None else tol / 8 * abs(C).max()

    if verbose:
        print("Regularization parameter: gamma = {}".format(gamma))
        print("Tolerance for duality gap : {:.2e}".format(tol / 2))
        print("Tolerance for ||Ax - b||_1: {:.2e}".format(tol / 2))

    a_org = a_dist.copy()
    b_org = b_dist.copy()
    a_dist = a_dist.reshape(-1, 1)
    b_dist = b_dist.reshape(-1, 1)
    a_dist = (1 - tol / 8) * a_dist + (tol / (8 * n)) * np.repeat(1, n).reshape(-1, 1)
    b_dist = (1 - tol / 8) * b_dist + (tol / (8 * n)) * np.repeat(1, n).reshape(-1, 1)

    inner_iters = inner_iters if inner_iters is not None else n

    np.random.seed(seed)

    tau2 = 1 / (2 * batch_size)
    y_store = np.zeros((n, 1), dtype=float)  # y_0
    z_temp = np.zeros((n, 1), dtype=float)  # z_0
    v_temp = np.zeros((n, 1), dtype=float)  # v_0
    v_tilde = np.zeros((n, 1), dtype=float)  # v_tilde_0

    logs = {
        "transport_matrix": []
    }

    C_temp = 0  # C_0
    D_temp = 0  # D_0

    # Lemma 1 + remark: Set Lipschitz constant
    L_bar = 1 / gamma if not mirror_descent else 5 / gamma

    x_s = np.zeros((n, n), dtype=float)  # x_0

    for s in range(1, num_iters + 1):
        tau_1s = 2 / (s + 4)

        alpha_s = 1 / (9 * tau_1s * L_bar)

        recorded_gradients = fullgrad(v=v_tilde, gamma=gamma, C=C, a_dist=a_dist, b_dist=b_dist, n=n)

        full_gradient = recorded_gradients.mean(axis=0).reshape(-1, 1)

        store_y = []
        for _ in range(inner_iters):
            # Pick random index(es)
            j = np.random.choice(n, batch_size, p=a_dist.reshape(-1))

            # Update v_temp
            v_temp = tau_1s * z_temp + tau2 * v_tilde + (1 - tau_1s - tau2) * y_store

            # Grad tilde
            grad_temp = full_gradient
            batch_grad = batchgrad(v=v_temp, gamma=gamma, C=C, a_dist=a_dist, b_dist=b_dist, idxes=j, n=n)
            diff = (batch_grad - recorded_gradients[j]).mean(axis=0).reshape(-1, 1)
            grad_temp += n * a_dist[j] * diff

            # Update z_temp
            z_temp = z_temp - alpha_s * grad_temp

            # Update y_store
            if mirror_descent:
                y_store = v_temp - np.linalg.norm(grad_temp, ord=1) / (9 * L_bar) * np.sign(grad_temp)
            else:
                y_store = v_temp - grad_temp / (9 * L_bar)
            store_y.append(y_store)

        store_y = np.array(store_y).reshape(-1, n)

        # Update v_tilde_{s+1}
        v_tilde = store_y.mean(axis=0).reshape(-1, 1)

        # Update C_s
        C_temp = C_temp + 1 / tau_1s

        # Pick y_tilde_s at random
        t = np.random.choice(inner_iters)
        random_y = store_y[t, :].reshape(-1, 1)

        # Update D_s
        D_temp = D_temp + (1 / tau_1s) * primal_semidual(random_y, a_dist=a_dist, b_dist=b_dist, C=C, n=n, gamma=gamma)

        # Update x_s
        x_s = (D_temp / C_temp).reshape(n, n).T

        # Check convergence
        error = abs(x_s.sum(axis=1).reshape(-1, 1) - a_dist).sum() + abs(x_s.sum(axis=0).reshape(-1, 1) - b_dist).sum()

        # Print status
        if verbose and s % print_every == 0:
            print("Iter = {:-5d} | ||Ax - b||_1 = {:.2e}".format(s, error))

        if save_iterates:
            transport_matrix = round_matrix(x_s, a_org, b_org)
            logs["transport_matrix"].append(transport_matrix)

        # Terminate if tolerance is achieved
        converged = error < tol / 2
        if check_termination and converged:
            if verbose:
                print("PDASMD converged after {} iterations".format(s))
            break
        if s >= num_iters and not converged:
            warnings.warn("WARNING: PDASMD did not converge after {} iterations"
                          .format(num_iters))

    x_s = round_matrix(x_s, a_org, b_org)

    return x_s, logs
