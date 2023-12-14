import numpy as np
import numpy.linalg as LA
from .round import round_matrix
import warnings

FLOAT_TYPE = np.double


def frobenius(X, Y):
    """
    Frobenius inner product of two arrays
    """
    X, Y = X.flatten(), Y.flatten()
    return np.dot(X, Y)


def objective(X, a, b):
    """
    Equality constraints
    """
    return np.linalg.norm(X.sum(1) - a, ord=1) \
           + np.linalg.norm(X.sum(0) - b, ord=1)


def recover(u, v, K):
    """
    From dual variables u and v, each of shape (n, ),
    recover the optimal transport matrix X.
    K is the Gibbs kernel matrix, of shape (n, n).
    """
    n = u.shape[0]
    assert v.shape == (n,), "v must be of shape ({},)".format(n)
    assert K.shape == (n, n), "K must be of shape ({}, {})".format(n, n)
    # return np.dot(np.dot(np.diag(u), K), np.diag(v))
    u = np.exp(u)
    v = np.exp(v)
    return v * (u * K.T).T


def round_matrix2(u, v, K, a, b):
    """
    Rounding step.
    u and v are the dual variables, each of shape (n, ).
    K is the Gibbs kernel matrix, of shape (n, n).
    a and b are the source and destination distributions, respectively.

    From Altschuler et al., 2017.
    Based on Peyre, "Computational Optimal Transport", Remark 4.6.
    """
    u_prime = u * np.minimum(1, a / (u * np.dot(K, v)))
    v_prime = v * np.minimum(1, b / (v * np.dot(K.T, u_prime)))
    delta_a = a - u_prime * np.dot(K, v_prime)
    delta_b = b - v_prime * np.dot(K.T, u_prime)
    X = np.dot(np.dot(np.diag(u_prime), K), np.diag(v_prime)) \
        + np.outer(delta_a, delta_b) / np.linalg.norm(delta_a, ord=1)
    return X


def sinkhorn(a,
             b,
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
    Sinkhorn algorithm.

    args:
        a: Source distribution. Non-neg array of shape (n, ), sums to 1.
        b: Destination distribution. Non-neg array of shape (n, ), sums to 1.
        C: Cost matrix. Non-neg array of shape (n, n)
        num_iters: Maximum number of Sinkhorn iterations. Int.
        verbose: Whether to print progress. Bool.
        print_every: Print progress every this many iterations. Int.
        tol: Primal optimality tolerance: output X s.t. f(X) <= f(X*) + tol.
             Float. This is epsilon in the algorithm.
        gamma: (Optional) Parameter for entropic regularization. If None, then
               gamma is set to tol / (4 * log(n)).
        check_termination: Whether to check termination condition. If True,
                           optimization is terminated when the Sinkhorn tolerance
                           is reached. If False, Sinkhorn is run for num_iters.
        save_iterates: Whether to save primal and dualsolutions for all iterations.
                       If False, only the final solutions are recorded. Can be memory
                       intensive for large n and large num_iters.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates

    Reference: Dvurechensky et al., 2018. Algorithm 1.
    """
    n = a.shape[0]
    assert b.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a) > 0, "Sinkhorn requires the source a to be positive"
    assert np.min(b) > 0, "Sinkhorn requires the destination b to be positive"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"

    a = a.astype(FLOAT_TYPE)
    b = b.astype(FLOAT_TYPE)
    C = C.astype(FLOAT_TYPE)

    ##### STEP 1: Set up gamma and Sinkhorn tolerance #####
    gamma = gamma if gamma is not None else tol / (4 * np.log(n))
    sinkhorn_tol = tol / (8 * np.max(C))
    if verbose:
        print("Regularization parameter: gamma = {:.2e}".format(gamma))
        print("Sinkhorn tolerance      : tol   = {:.2e}".format(sinkhorn_tol / 2))

    ##### STEP 2: Rescale the marginals #####
    a_org = a.copy()
    b_org = b.copy()
    a = (1 - sinkhorn_tol / 8) * \
        (a + (sinkhorn_tol / (n * (8 - sinkhorn_tol))))
    b = (1 - sinkhorn_tol / 8) * \
        (b + (sinkhorn_tol / (n * (8 - sinkhorn_tol))))

    assert np.allclose(a.sum(), 1)
    assert np.allclose(b.sum(), 1)

    # Gibbs kernel
    K = np.exp(- C / gamma)

    # Initialization
    u = np.zeros(n, dtype=FLOAT_TYPE)
    v = np.zeros(n, dtype=FLOAT_TYPE)
    X = recover(u, v, K)
    obj = objective(X, a, b)

    logs = {
        "objective": [obj],
        "u": [u],
        "v": [v],
        "X": [X],
        "CX": [],
        "transport_matrix": []
    }

    ##### STEP 3: Run Sinkhorn until (sinkhorn / 2) accuracy #####
    for k in range(1, num_iters + 1):
        u = logs["u"][-1]
        v = logs["v"][-1]

        # Sinkhorn iterates: update dual variables
        if k % 2 == 0:
            u = u + np.log(a) - np.log(X.sum(1))
            if save_iterates:
                logs["u"].append(u)
            else:
                logs["u"] = [u]
        else:
            v = v + np.log(b) - np.log(X.sum(0))
            if save_iterates:
                logs["v"].append(v)
            else:
                logs["v"] = [v]

        # Recover primal variables
        X = recover(logs["u"][-1], logs["v"][-1], K)
        obj = objective(X, a, b)
        logs["objective"].append(obj)
        if save_iterates:
            logs["X"].append(X)
        else:
            logs["X"] = [X]

        if save_iterates:
            transport_matrix = round_matrix(X, a_org, b_org)
            logs["transport_matrix"].append(transport_matrix)
            logs["CX"].append(np.sum(C * transport_matrix))

        # Print progress
        if verbose and k % print_every == 0:
            print("Iter = {:-5d} | "
                  "Loss = {:.2e}"
                  .format(k, logs["objective"][-1]))

        # Check if converged
        converged = np.abs(logs["objective"][-1]) < sinkhorn_tol / 2
        if check_termination and converged:
            if verbose:
                print("Sinkhorn converged after {} iterations".format(k))
            break
        if k >= num_iters and not converged:
            warnings.warn("WARNING: Sinkhorn did not converge after {} iterations"
                          .format(num_iters))

    #### STEP 4: Round the primal solution #####
    X = round_matrix(X=logs["X"][-1].reshape(n, n), a=a_org, b=b_org)

    return X, logs
