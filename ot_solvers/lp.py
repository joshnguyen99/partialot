import cvxpy as cp
import numpy as np


def lp(a,
       b,
       C,
       verbose=True,
       tol=1e-6):
    """
    Solving OT as a linear program.

    args:
        a: Source distribution. Non-neg array of shape (n, ), sums to 1.
        b: Destination distribution. Non-neg array of shape (n, ), sums to 1.
        C: Cost matrix. Non-neg array of shape (n, n)
        verbose: Whether to print progress. Bool.
        tol: Suboptimality toleraance. Depends on the solver.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (n, n).
    """

    n = a.shape[0]
    assert b.shape == (n,), "b must be of shape ({},)".format(n)
    assert np.min(a) > 0, "The source distribution must be non-negative"
    assert np.min(b) > 0, "The target distribution must be non-negative"
    assert C.shape == (n, n), "C must be of shape ({}, {})".format(n, n)
    assert np.alltrue(C >= 0), "The cost matrix must be non-negative"
    assert tol > 0, "tol must be positive"

    # Optimization variables: the transport matrix
    x = cp.Variable(n * n)

    # Objective function: total transport cost
    objective = cp.Minimize(C.flatten() @ x)

    # Constraints
    constraints = [x >= 0]
    # Column constraints
    for i in range(n):
        constraints.append(cp.sum(x[n * i:n * i + n:]) == a[i])
    # Row constraints
    for j in range(n):
        constraints.append(cp.sum(x[j::n]) == b[j])

    # Solve problem
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve(solver="GUROBI", verbose=verbose, BarConvTol=tol)

    return x.value.reshape(n, n)
