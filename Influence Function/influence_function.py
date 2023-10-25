from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linear(x, slope, intercept):
    return slope*x + intercept

def cost(func, params, xdata, ydata):
    squared_error = func(xdata, *params) - ydata
    return sum(squared_error)

# Influence functions are defined as the first-order Taylor approximation of this around epsilon = 0
# Or maybe the derivative if it's more convenient to subtract off the value at 0, I think I've seen both definitions
def response_function(new, epsilon, func, xdata, ydata):
    if epsilon == 0:
        params, _ = curve_fit(func, xdata, ydata)
        return params
    else:
        newxdata = np.concatenate((np.array([new[0]]), xdata))
        newydata = np.concatenate((np.array([new[1]]), ydata))
        weights = np.array([epsilon] + [1 for _ in xdata])
        params, _ = curve_fit(func, newxdata, newydata, sigma=1.0/weights)
        return params

def order_by_cost_influence(epsilon, func, xdata, ydata, cost):
    influences = []
    for i in range(len(xdata)):
        params0 = response_function((xdata[i],ydata[i]), 0, func, xdata, ydata)
        params1 = response_function((xdata[i],ydata[i]), epsilon, func, xdata, ydata)
        cost0 = cost(func, params0, xdata, ydata)
        cost1 = cost(func, params1, xdata, ydata)
        influences += [(cost1 - cost0)/epsilon]
    data = [((xdata[i], ydata[i]), influences[i]) for i in range(len(xdata))]
    return sorted(data, key = lambda x: x[1])
        

xdata = np.array([0.41,0.47,0.51,0.55,0.59,0.66,0.70])
ydata = np.array([0.11557,0.13557,0.15557 ,0.17557 ,0.19557 ,0.21557,0.23557])

print(order_by_cost_influence(1, linear, xdata, ydata, cost))
print(order_by_cost_influence(0.1, linear, xdata, ydata, cost))
print(order_by_cost_influence(0.01, linear, xdata, ydata, cost))
print(order_by_cost_influence(0.001, linear, xdata, ydata, cost))
plt.scatter(xdata,ydata)
plt.show()