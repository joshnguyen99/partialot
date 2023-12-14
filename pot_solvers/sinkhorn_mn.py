import numpy as np
import numpy.linalg as LA
import warnings
import time


def objective(X, a, b):
    """
    Equality constraints
    """
    return np.linalg.norm(X.sum(1) - a, ord=1) \
        + np.linalg.norm(X.sum(0) - b, ord=1)


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

    # assert np.alltrue(Xbar >= 0)
    # assert np.alltrue(pbar >= 0)
    # assert np.alltrue(qbar >= 0)
    # assert np.allclose(np.sum(Xbar), s)
    # assert np.allclose(Xbar.sum(1) + pbar, r)
    # assert np.allclose(Xbar.sum(0) + qbar, c)

    return Xbar, pbar, qbar


def round_matrix_ot(X, a, b):
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


def recover(u, v, K):
    """
    From dual variables u and v, of shapes (m,) and (n,) respectively
    recover the optimal transport matrix X.
    K is the Gibbs kernel matrix, of shape (m, n).
    """
    m, n = K.shape
    # return np.dot(np.dot(np.diag(u), K), np.diag(v))
    u = np.exp(u)
    v = np.exp(v)
    return v * (u * K.T).T


def round_matrix_feasible(X, r, c, s):
    # Extract X, p, q from X_aug
    m = X.shape[0] - 1
    n = X.shape[1] - 1
    X_ = X[:m, :n]
    p_ = X[:m, -1]
    q_ = X[-1, :n]

    # Concatenate to (m * n + m + n) vector
    x = np.concatenate((X_.flatten(), p_, q_))

    # Round
    return round_matrix(x, r, c, s)


def sinkhorn_mn(a,
                b,
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
                feasible=True,
                A_mult=1.1,
                ):
    """
    Sinkhorn algorithm for POT.

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
        round: Whether to applying rounding to the final POT matrix.
        A_mult: Multiplicative constant C_max, must be > 1.0.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
        logs: dictionary of iterates

    Reference: Dvurechensky et al., 2018. Algorithm 1.
    """
    m, n = a.shape[0], b.shape[0]
    m_org, n_org = m, n
    assert np.min(a) > 0, "Sinkhorn requires the source a to be positive"
    assert np.min(b) > 0, "Sinkhorn requires the destination b to be positive"
    assert s <= min(a.sum(), b.sum()), \
        "s must be less than or equal to the sum of a and b"
    assert C.shape == (m, n), "C must be of shape ({}, {})".format(n, n)
    assert num_iters > 0, "num_iters must be positive"
    assert print_every > 0, "print_every must be positive"
    assert tol > 0, "tol must be positive"
    # assert A_mult > 1, "A / C_max must be > 1"

    # Original values
    a_org = a.copy()
    b_org = b.copy()
    s_org = s
    C_org = C.copy()
    m_org = a_org.shape[0]
    n_org = b_org.shape[0]

    # Augment the marginals and cost matrix
    a = np.append(a, np.sum(b) - s)
    b = np.append(b, np.sum(a[:m]) - s)
    maxC = np.max(C)
    C = np.append(C, np.zeros((m, 1)), axis=1)
    C = np.append(C, np.zeros((1, n + 1)), axis=0)
    # This is A in Chapel et al.
    C[-1, -1] = maxC * A_mult
    # C /= C.max()
    a_aug_org = a.copy()
    b_aug_org = b.copy()

    # Check shapes
    assert a.shape == (m + 1,)
    assert b.shape == (n + 1,)
    assert C.shape == (m + 1, n + 1)
    m = m + 1
    n = n + 1

    ##### STEP 1: Set up gamma and Sinkhorn tolerance #####
    gamma = gamma if gamma is not None else tol / (4 * np.log(max(m, n)))
    sinkhorn_tol = tol / (8 * np.max(C))
    if verbose:
        print("Regularization parameter: gamma = {:.2e}".format(gamma))
        print("Sinkhorn tolerance      : tol   = {:.2e}".format(sinkhorn_tol / 2))

    ##### STEP 2: Rescale the marginals #####
    a = (1 - sinkhorn_tol / 8) * \
        (a + (sinkhorn_tol / (n * (8 - sinkhorn_tol))))
    b = (1 - sinkhorn_tol / 8) * \
        (b + (sinkhorn_tol / (n * (8 - sinkhorn_tol))))

    # Gibbs kernel
    K = np.exp(- C / gamma)

    # Initialization
    u = np.zeros(m)
    v = np.zeros(n)
    X = recover(u, v, K)
    obj = objective(X, a, b)

    logs = {
        # Variables in Sinkhorn (m+1, n+1)
        "objective": [obj],
        "u": [u],
        "v": [v],
        "X": [X],
        "row_cons_err": [],
        "col_cons_err": [],
        "total_mass_err": [],
        "time_per_iter": [],
        "CX_unrounded": [],

        # Variables after naive rounding
        "X_rounded_ot": [],
        "CX_rounded_ot": [],
        "A_ot": [],  # Total mass error for Altschuler rounded matrix

        # Variables after feasible rounding
        "X_rounded_feasible": [],
        "CX_rounded_feasible": [],
    }

    start_time = old_time = new_time = time.time()

    ##### STEP 3: Run Sinkhorn until (sinkhorn / 2) accuracy #####
    for k in range(1, num_iters + 1):
        u = logs["u"][-1]
        v = logs["v"][-1]

        old_time = time.time()

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
            new_time = time.time()
            logs["time_per_iter"].append(new_time - old_time)

            # Update logs
            logs["X"].append(X)
            logs["row_cons_err"].append(
                np.linalg.norm(X.sum(1)[:-1] - a[:-1], ord=2)
            )
            logs["col_cons_err"].append(
                np.linalg.norm(X.sum(0)[:-1] - b[:-1], ord=2)
            )
            logs["total_mass_err"].append(
                abs(X[:-1, :-1].sum() - s)
            )

            if round:
                # No rounding
                CX = np.sum(C_org * X[:-1, :-1])
                logs["CX_unrounded"].append(CX)

                # Naive rounding (Altschuler et al.)
                X_rounded_ot = round_matrix_ot(X.copy(), a_aug_org, b_aug_org)
                X_rounded_ot = X_rounded_ot[:-1, :-1]
                ot_total_mass_err = abs(X_rounded_ot.sum() - s_org)

                CX_rounded_ot = np.sum(C_org * X_rounded_ot)
                logs["X_rounded_ot"].append(X_rounded_ot)
                logs["CX_rounded_ot"].append(CX_rounded_ot)
                A_ot = abs(X_rounded_ot.sum() - s_org)
                logs["A_ot"].append(A_ot)

                # Feasible rounding (Altschuler + our method)
                X_rounded_feasible, p_rounded_feasible, q_rounded_feasible = round_matrix_feasible(X.copy(), a_org,
                                                                                                   b_org, s_org)
                CX_rounded_feasible = np.sum(C_org * X_rounded_feasible)
                logs["X_rounded_feasible"].append(X_rounded_feasible)
                logs["CX_rounded_feasible"].append(CX_rounded_feasible)
                pot_total_mass_err = abs(X_rounded_feasible.sum() - s_org)
                pot_row_mass_err = np.sum(np.abs(X_rounded_feasible.sum(1) + p_rounded_feasible - a_org))
                pot_col_mass_err = np.sum(np.abs(X_rounded_feasible.sum(0) + q_rounded_feasible - b_org))

            # if f_star is not None and check_termination is True and CX - f_star < tol:
            #     break

        else:
            logs["X"] = [X]

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
    running_time = time.time() - start_time
    logs["time"] = running_time
    if round:
        if feasible:
            # Feasible rounding
            X = logs["X"][-1]
            X = round_matrix_feasible(X[:, :], a_org, b_org, s_org)[0]
        else:
            # Naive rounding
            X = logs["X"][-1]
            X = round_matrix_ot(X, a_aug_org, b_aug_org)
            X = X[:-1, :-1]
    else:
        X = logs["X"][-1].reshape(m, n)[:-1, :-1]

    return X, logs
