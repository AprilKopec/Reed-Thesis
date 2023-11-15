import numpy as np

def linear(x, slope, intercept):
    return slope*x + intercept

model = linear

def squared_error(params, x, y):
    squared_error = (model(x, *params) - y)**2
    return squared_error

loss = squared_error

xdata = np.array([5, 8, 7, 3, 6, 4, 1, 9])
ydata = np.array([1, 1, 1, 0, 1, 0, 0, 1])

# Given a function F:Theta->R and points p, q in Theta,
# the Bregman divergence gives the difference between F(p)
# and the first-order taylor approximation of F(p) around q
def bregman_divergence(F, p, q):
    d: float = 2**-16 # This can be changed if we need more precision

    dist = np.linalg.norm(p-q)
    dir = (p-q) / dist

    dx = dir*d
    dir_derivative = (F(q+dx/2)-F(q-dx/2))/d

    return F(p) - (F(q) + dir_derivative*dist)

final_params = np.array([1, 2])

# This is just directly copied from the Anthropic paper's
# definition of the proximal Bregman objective.
def proximal_bregman_objective(params, new_data, epsilon):
    damping: float = 2**-8 # "lambda" is a reserved code word :|
    N = len(xdata)

    term1 = 0
    for i in range(N):
        Li = lambda params: loss(params, xdata[i], ydata[i])
        term1 += bregman_divergence(Li, params, final_params)
    term1 /= N

    term2 = epsilon * loss(params, new_data[0], new_data[1])
    term3 = (damping/2) * np.linalg.norm(params - final_params)

    return term1 + term2 + term3