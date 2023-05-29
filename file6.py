import numpy as np
import scipy as sp

#minimum

bounds = sp.optimize.Bounds ([0, 0], [np.inf, np.inf])
linear_constraint = sp.optimize.LinearConstraint ([1, 1], -np.inf, 6)

def con(x):
    return (x[0] -2)*(x[1]+1)

nonlinear_constraint = sp.optimize.NonlinearConstraint(con, -np.inf, 4)

x0 = np.array([0, 0])

def fun(x):
    return x[0] + x[1]

res = sp.optimize.minimize(fun, x0, method='trust-constr',
                constraints=[linear_constraint, nonlinear_constraint],
                options={'verbose': 1}, bounds=bounds)


print("Minimum: ")
print(res.x[0] + res.x[1])

#maximum

bounds = sp.optimize.Bounds ([0, 0], [np.inf, np.inf])
linear_constraint = sp.optimize.LinearConstraint ([1, 1], -np.inf, 6)

def con(x):
    return -(x[0] -2)*(x[1]+1)

nonlinear_constraint = sp.optimize.NonlinearConstraint(con, -np.inf, 4)

x0 = np.array([0, 0])

def fun(x):
    return -(x[0] + x[1])

res = sp.optimize.minimize(fun, x0, method='trust-constr',
                constraints=[linear_constraint, nonlinear_constraint],
                options={'verbose': 1}, bounds=bounds)

print("Maximum: ")

print(res.x[0] + res.x[1])