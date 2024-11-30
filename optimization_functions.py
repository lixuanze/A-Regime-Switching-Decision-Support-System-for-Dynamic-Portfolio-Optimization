import cvxpy as cp
import numpy as np
from scipy.stats.distributions import chi2


def simple_mvo(mu, Sigma, R, C, b):
    n = len(mu)

    # Variables
    x = cp.Variable(n)

    # Objective: Minimize risk
    risk = cp.quad_form(x, Sigma)

    # Constraints
    constraints = [cp.sum(x) == 1, x >= 0, C @ x == b, mu.T @ x >= R]

    # Problem
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    return x.value


def constrained_mvo(mu, Sigma, lm):
    n = len(mu)

    # Variables
    x = cp.Variable(n)

    # Objective: Minimize risk - return
    risk = lm * cp.quad_form(x, Sigma)
    ret = mu.T @ x

    # Constraints
    constraints = [cp.sum(x) == 1, x >= 0]

    # Problem
    prob = cp.Problem(cp.Minimize(risk - ret), constraints)
    prob.solve()

    return x.value


def robust_mvo(mu, Sigma, lambda_, alpha, T, C, b, ub=1, lb=0, cardinality=False, k=10, transaction_penalty=0, x_last=None, include_trans_cost=False, k_stock=20):
    n = len(mu)

    # Uncertainty set
    epsilon = np.sqrt(chi2.ppf(alpha, df=n))
    var = np.diag(Sigma)
    st_dev = np.sqrt(var)
    theta_half = 1 / np.sqrt(T) * np.diag(st_dev)

    # Variables
    x = cp.Variable(n)
    norm = cp.Variable()

    # Objective: Minimize risk - return + norm
    risk = lambda_ * cp.quad_form(x, Sigma)
    ret = mu.T @ x

    # Norm constraint for robust optimization
    diff = epsilon ** 2 * theta_half @ x

    # Objective
    objective = cp.Minimize(risk - ret + norm)

    # Constraints
    constraints = [cp.sum(x) == 1, x >= 0, C @ x == b, norm >= cp.norm(diff, 2), x <= ub, x >= lb]

    if cardinality:
        y = cp.Variable(n, boolean=True)
        constraints += [y <= x, x <= ub * y, cp.sum(y) <= k_stock]

    if include_trans_cost:
        z = cp.Variable(n)
        constraints += [x - x_last <= z, x_last - x <= z]
        objective = cp.Minimize(risk - ret + norm + transaction_penalty * cp.sum(z))

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value


def cvar(alpha, returns, C, b, cardinality=False, ub=1, lb=0, k_stock=10, transaction_penalty=0.01, x_last=None, include_trans_cost=False):
    n = returns.shape[1]
    S = returns.shape[0]
    
    # Variables
    x = cp.Variable(n)
    gamma = cp.Variable()
    z = cp.Variable(S, nonneg=True)

    # Objective: Minimize CVaR
    losses = -returns.to_numpy() @ x
    objective = cp.Minimize(gamma + 1 / ((1 - alpha) * S) * cp.sum(z))
    # Constraints
    constraints = [z >= losses - gamma, cp.sum(x) == 1, x >= lb, x <= ub, C @ x == b]
    if cardinality:
        y = cp.Variable(n, boolean=True)
        constraints += [y <= x, x <= ub * y, cp.sum(y) <= k_stock]

    if include_trans_cost:
        v = cp.Variable(n)
        constraints += [x - x_last <= v, x_last - x <= v]
        objective = cp.Minimize(gamma + 1 / ((1 - alpha) * S) * cp.sum(z) + transaction_penalty * cp.sum(v))

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value


def sharpe_ratio(mu, Sigma, rf, ub=1, lb=0):
    n = len(mu)

    # Risk-free rate vector
    rf = rf * np.ones(n)

    # Variables
    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)

    # Objective: Maximize Sharpe Ratio (Minimize variance for a given return)
    risk = cp.quad_form(y, Sigma)
    excess_ret = (mu - rf).T @ y

    # Constraints
    constraints = [excess_ret == 1, cp.sum(y) == kappa, y >= lb * kappa, y <= ub * kappa]

    # Problem
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    x = y.value / sum(y.value)
    return x.flatten()


def robust_sharpe_ratio(mu, Sigma, rf, alpha, T, C, b, ub=1, lb=0, cardinality=False, k_stock=20, transaction_penalty=0, include_trans_cost=False, x_last=None):
    n = len(mu)
    epsilon = np.sqrt(chi2.ppf(alpha, df=n))

    # Uncertainty set
    var = np.diag(Sigma)
    st_dev = np.sqrt(var)
    theta_half = 1 / np.sqrt(T) * np.diag(st_dev)

    # Risk-free rate vector
    rf = rf * np.ones(n)

    # Variables
    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    norm = cp.Variable()

    # Objective: Maximize Robust Sharpe Ratio
    risk = cp.quad_form(y, Sigma)
    excess_ret = (mu - rf).T @ y

    # Constraints
    diff = epsilon ** 2 * theta_half @ y
    constraints = [excess_ret - norm >= 1, cp.sum(y) >= 0, y >= lb * kappa, y <= ub * kappa, C @ y == b * kappa]

    if cardinality:
        z = cp.Variable(n, boolean=True)
        constraints += [y <= ub * z, y >= lb * z, cp.sum(z) <= k_stock]

    if include_trans_cost:
        u = cp.Variable(n)
        constraints += [y - x_last <= u, x_last - y <= u]
        objective = cp.Minimize(risk + transaction_penalty * cp.sum(u))
    else:
        objective = cp.Minimize(risk)

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    x = y.value / sum(y.value)
    return x.flatten()


def risk_parity(Sigma, C, b):
    n = Sigma.shape[0]

    # Variables
    x = cp.Variable(n)
    theta = cp.Variable()
    z = cp.Variable(n)

    # Objective: Minimize the sum of z
    objective = cp.Minimize(cp.sum(z))

    # Constraints
    constraints = [cp.sum(x) == 1, C @ x == b]
    for i in range(n):
        constraints.append(x[i] * x[i] * (Sigma[i, :] @ x)**2 - 2 * x[i] * (Sigma[i, :] @ x) * theta + theta**2 >= z[i])

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value


def minimum_variance(Sigma, C, b, ub=1, lb=0, cardinality=False, k_stock=20, transaction_penalty=0, x_last=None, include_trans_cost=False):
    n = Sigma.shape[0]

    # Variables
    x = cp.Variable(n)

    # Objective: Minimize variance
    risk = cp.quad_form(x, Sigma)
    objective = cp.Minimize(risk)

    # Constraints
    constraints = [cp.sum(x) == 1, x >= lb, x <= ub, C @ x == b]

    if cardinality:
        y = cp.Variable(n, boolean=True)
        constraints += [y <= x, x <= ub * y, cp.sum(y) <= k_stock]

    if include_trans_cost:
        z = cp.Variable(n)
        constraints += [x - x_last <= z, x_last - x <= z]
        objective = cp.Minimize(risk + transaction_penalty * cp.sum(z))

    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value
