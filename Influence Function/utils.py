import numpy as np

differential_length = 2**-16

def gradient(f, v):
    n = len(v)
    grad = np.zeros(n)
    for i in range(len(v)):
        d = np.zeros(n)
        d[i] = differential_length
        directional_derivative = (f(v+d/2) - f(v-d/2))/differential_length
        grad[i] = directional_derivative
    return grad

# Takes a function f:R^n -> R^m and returns an
# m by n matrix J where J[i][j] = df_i / Dx_j
# Then, f(x_0) + J(x-x_0) is the best linear approximation
# of f(x) around x_0

# I was writing this for loss influence but actually I can just use a gradient for that
def vector_derivative(f, x):
    n = len(x)
    m = len(f(x))
    J = np.array([[0 for _ in range(n)] for _ in range(m)])
    print(J)
    for i in range(m):
        fi = lambda x: [f(x)][i]
        J[i] = gradient(fi, x)
    return J