import numpy as np
import numpy.linalg as LA
from scipy.special import xlogy
from .round import round_matrix
import warnings
import time


def bregman_proj(p1,
                 p2,
                 C,
                 s,
                 num_iters=10000,
                 verbose=False,
                 print_every=1000,
                 tol=1e-5,
                 gamma=None,
                 check_termination=True,
                 save_iterates=True,
                 cost_old=10000,
                 ):
    """
    Scaling algorithm for POT.

    args:
        a: Source distribution. Non-neg array of shape (m, ).
        b: Destination distribution. Non-neg array of shape (n, ).
        C: Cost matrix. Non-neg array of shape (m, n)
        s: total mass to transport. Must be less than equal to
           sum(a_dist) and sum(b_dist).
        num_iters: Maximum number of Sinkhorn iterations. Int.
        verbose: Whether to print progress. Bool.
        print_every: Print progress every this many iterations. Int.
        tol: Primal optimality tolerance: output X s.t. f(X) <= f(X*) + tol.
             Float. This is epsilon in the algorithm.
        gamma: (Optional) Parameter for entropic regularization. If None, then
               gamma is set to tol / (4 * log(n + 1)).
        check_termination: Whether to check termination condition. If True,
                           optimization is terminated when the Sinkhorn tolerance
                           is reached. If False, Sinkhorn is run for num_iters.
        save_iterates: Whether to save primal and dualsolutions for all iterations.
                       If False, only the final solutions are recorded. Can be memory
                       intensive for large n and large num_iters.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates
    """

    m, n = p1.shape[0], p2.shape[0]
    assert np.min(p1) > 0, "Scaling requires the source a to be positive"
    assert np.min(p2) > 0, "Scaling requires the destination b to be positive"
    assert s <= min(p1.sum(), p2.sum()), \
        "s must be less than or equal to the sum of a and b"

    b = np.ones(n)
    K = np.exp(-C / gamma)

    tol = 1e-6
    max_iter = 500
    z = 1
    for i in range(max_iter):
        s1 = z * (K @ b + 1e-16)
        a = np.minimum(p1, np.maximum(0, s1)) / s1
        s2 = z * (K.T @ a + 1e-16)
        b = np.minimum(p2, np.maximum(0, s2)) / s2
        s3 = a.T @ K @ b + 1e-16
        z = np.minimum(s, np.maximum(0, s3)) / s3

        if i % 20 == 0:
            X = z * (b * (a * K.T).T)
            cost = (X * C).sum()
            if np.abs(cost - cost_old) < tol:
                break

    return X, cost
