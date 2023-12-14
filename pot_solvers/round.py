import numpy as np
import numpy.linalg as LA


def extract(x, m, n):
    X = x[:m * n].reshape(m, n)
    p = x[m * n:m * n + m]
    q = x[m * n + m:]
    return X, p, q


def threshold_zero(x):
    x = np.maximum(x, 0)
    return np.where(np.abs(x) > 1e-15, x, np.zeros_like(x))


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


def round_matrix(x, r, c, s):
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

    # assert np.alltrue(Xbar >= 0)
    # assert np.alltrue(pbar >= 0)
    # assert np.alltrue(qbar >= 0)
    # assert np.allclose(np.sum(Xbar), s)
    # assert np.allclose(Xbar.sum(1) + pbar, r)
    # assert np.allclose(Xbar.sum(0) + qbar, c)

    return Xbar, pbar, qbar


def round_matrix_sinkhorn(X, a, b):
    """
    Given approximate primal solution X and marginals a and b,
    project X onto the feasible set U_r(a, b).

    Copied from ot_solvers/round.py

    From Altschuler et al., 2017.
    """
    one_m = np.ones(X.shape[0])
    one_n = np.ones(X.shape[1])

    x = a / np.dot(X, one_n)
    x = np.minimum(x, 1)
    F_1 = X.T * x

    y = b / np.dot(F_1, one_m)
    y = np.minimum(y, 1)
    F_2 = (F_1.T * y).T

    err_a = a - F_2.T @ one_n
    err_b = b - F_2 @ one_m

    X_hat = F_2.T + np.outer(err_a, err_b) / LA.norm(err_a, ord=1)

    assert np.allclose(X_hat.sum(1), a)
    assert np.allclose(X_hat.sum(0), b)

    # This must hold
    # lhs = LA.norm(X_hat - X, ord=1)
    # rhs = 2 * (LA.norm(X.sum(1) - a, ord=1) + LA.norm(X.sum(0) - b, ord=1))
    # assert lhs <= rhs

    return X_hat
