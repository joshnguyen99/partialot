import cvxpy as cp
import numpy as np


def lp(r,
       c,
       C,
       s,
       verbose=True,
       tol=1e-6):
    """
    Solving OT as a linear program.

    args:
        r: Source distribution. Non-neg array of shape (m, ), sums to 1.
        c: Destination distribution. Non-neg array of shape (n, ), sums to 1.
        C: Cost matrix. Non-neg array of shape (m, n)
        s: Total mass to transport. Positive scalar, must be less than or equal to
              the total mass of r and of c.
        verbose: Whether to print progress. Bool.
        tol: Suboptimality toleraance. Depends on the solver.

    returns:
        X: Optimal transport matrix. Non-neg array of shape (m, n).
    """

    m, n = r.shape[0], c.shape[0]
    assert np.min(r) >= 0, "The source distribution must be non-negative"
    assert np.min(c) >= 0, "The target distribution must be non-negative"
    assert C.shape == (m, n), "C must be of shape ({}, {})".format(n, n)
    assert np.alltrue(C >= 0), "The cost matrix must be non-negative"
    assert s > 0, "Amount of transport s must be positive"
    assert s <= np.sum(r), "Amount of transport s must be less than or equal to the total mass of r"
    assert s <= np.sum(c), "Amount of transport s must be less than or equal to the total mass of c"
    assert tol > 0, "tol must be positive"

    # Optimization variables: the transport matrix
    x = cp.Variable(m * n)

    # Objective function: total transport cost
    objective = cp.Minimize(C.flatten() @ x)

    # Constraints
    constraints = [x >= 0]
    # Column constraints
    for i in range(m):
        constraints.append(cp.sum(x[n * i:n * i + n:]) <= r[i])
    # Row constraints
    for j in range(n):
        constraints.append(cp.sum(x[j::n]) <= c[j])
    # Total trasnport mass constraint
    constraints.append(cp.sum(x) == s)

    # Solve problem
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve(solver="GUROBI", verbose=verbose, BarConvTol=tol)

    return x.value.reshape(m, n)
