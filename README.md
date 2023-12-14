# Partial Optimal Transport

This repository contains implementation of algorithms for solving the (partial) optimal transport plan between two discrete distributions. Check out our [AAAI 2024](https://aaai.org/aaai-conference/) paper below.

Anh Duc Nguyen, Tuan Dung Nguyen, Minh Quang Nguyen, Hoang H. Nguyen, Lam M. Nguyen, and Kim-Chuan Toh. **"On Partial Optimal Transport: Revising the Infeasibility of Sinkhorn and Efficient
Gradient Methods"**. In: *Proceedings of the AAAI Conference on Artificial Intelligence (to appear)* 38 (2024)

## (Partial) Optimal Transport

Suppose we have two marginal distributions $\boldsymbol{r} \in \mathbb{R}^m$ and $\boldsymbol{c} \in \mathbb{R}^{n}$ and a non-negative cost matrix $\boldsymbol{C} \in \mathbb{R}_{+}^{m \times n}$. If the total masses in the marginals are equal, then we have the optimal transport problem
$$
\begin{align*}
    \text{\textbf{OT}}(\boldsymbol{r}, \boldsymbol{c}) = &\argmin_{\boldsymbol{X} \in \mathbb{R}_{+}^{m \times n}}  \left< \boldsymbol{C}, \boldsymbol{X} \right>_{F} \\
    &\text{subject to} ~ \boldsymbol{X} \boldsymbol{1}_n = \boldsymbol{r} ~\text{and}~ \boldsymbol{X}^\top \boldsymbol{1}_m = \boldsymbol{c}.
\end{align*}
$$

If the total masses are not equal, we can only transport at most $s = \min\{ \| \boldsymbol{r} \|_1, \| \boldsymbol{c} \|_1\}$ amount of mass in total. This leads us to the following partial optimal transport problem
$$
\begin{align*}
    \text{\textbf{POT}}(\boldsymbol{r}, \boldsymbol{c}, s) = &\argmin_{\boldsymbol{X} \in \mathbb{R}_{+}^{m \times n}}  \left< \boldsymbol{C}, \boldsymbol{X} \right>_{F} \\
    &\text{subject to} ~ \boldsymbol{X} \boldsymbol{1}_n \leq \boldsymbol{r}, ~ \boldsymbol{X}^\top \boldsymbol{1}_m \leq \boldsymbol{c} ~\text{and}~ \boldsymbol{1}_m^\top \boldsymbol{X} \boldsymbol{1}_n = s.
\end{align*}
$$

The goal is to find an approximate solution to **POT** efficiently. In particular, given an error tolerance $\varepsilon \geq 0$, we want to find a feasible solution $\boldsymbol{X}$ such that $\left< \boldsymbol{C}, \boldsymbol{X} \right>_{F} \leq \left< \boldsymbol{C}, \text{\textbf{POT}}(\boldsymbol{r}, \boldsymbol{c}, s) \right>_{F} + \varepsilon$ (cf. Definition 1).

This repository contains implementation of two algorithms for finding $\varepsilon$-approximate solutions presented in our paper:
- Adaptive Primal-Dual Accelerated Gradient Descent (APDAGD), with time complexity  $\widetilde{\mathcal{O}}(n^{2.5} / \varepsilon)$; and
- Dual Extrapolation (DE), with time complexity $\widetilde{\mathcal{O}}(n^{2} / \varepsilon)$.

## Create a Python Environment

First, create an environment.
```bash
conda create --name partialot python=3.9.12
```

Then install the required packages.
```bash
pip3 install -r requirements.txt
```

We will be using Gurobi as the linear program solver in the backend via [`cvxpy`](https://www.cvxpy.org). You can download this solver [here](https://www.gurobi.com/downloads/); the license is available for free for academic purposes. If you prefer another solver, modify it in [`ot_solvers/lp.py`](./ot_solvers/lp.py) and [`pot_solvers/lp.py`](./pot_solvers/lp.py).