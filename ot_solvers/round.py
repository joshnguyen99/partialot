import numpy as np
import numpy.linalg as LA


def round_matrix(X, a, b):
    """
    Given approximate primal solution X and marginals a and b,
    project X onto the feasible set U_r(a, b).

    From Altschuler et al., 2017.
    """
    one = np.ones(X.shape[0])

    x = a / np.dot(X, one)
    x = np.minimum(x, 1)
    F_1 = X.T * x

    y = b / np.dot(F_1.T, one)
    y = np.minimum(y, 1)
    F_2 = (F_1 * y).T

    err_a = a - F_2 @ one
    err_b = b - F_2.T @ one

    X_hat = F_2 + np.outer(err_a, err_b) / LA.norm(err_a, ord=1)

    assert np.allclose(X_hat.sum(1), a)
    assert np.allclose(X_hat.sum(0), b)

    # This must hold
    # lhs = LA.norm(X_hat - X, ord=1)
    # rhs = 2 * (LA.norm(X.sum(1) - a, ord=1) + LA.norm(X.sum(0) - b, ord=1))
    # assert lhs <= rhs

    return X_hat
