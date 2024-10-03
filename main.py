import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# following tutorial
# https://www.youtube.com/watch?v=MM3cBamj1Ms
#
# Book if you need help
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html



# coupled first_order ODE
#y1' = y1 + y2^2 + 3x
#y2' = 3y1 + y2^3 - cos(x)
# y1(0) = (0)
# y2(0) = 0
# x is our independent variable (kinda like t)

# S = (y1, y2)
# dS/dx = (dy1/dx, dy2/dx)
# just want our vector from its derivative!

# def dSdx(x, S):
#     y1, y2 = S
#     return [y1 + y2**2 + 3*x,
#             3*y1 + y2**3 - np.cos(x)]
# y1_0 = 0
# y2_0 = 0
# S_0 = (y1_0, y2_0)


# ----------

# Second Order ODE
# say x'' = -x'^2 + sin(x)
#
# We will convert 2nd order ODE to first order!
# x' = v
# v' = -v^2 + sin(x)
# boom! we've got two equations now!
# We know S = (x', v') = (v, -v^2 + sin(x))
def dSdx(x, S):
    x, v = S
    return [v,
            -v**2 + np.sin(x)]

x_0 = 2
v_0 = 5
S_0 = (x_0, v_0)


def dSdt(t, S):
    x1, x2 = S
    k1 = 2
    k2 = -1.9
    return [(-1 * x1) - (k1*x2),
            k2*x1 - x1**2 - x2]
x1_0 = 1
x2_0 = 1
S_0 = (x1_0, x2_0)





if __name__ == "__main__":
    # setup what x will be!
    # x = np.linspace(0, 1, 100)
    t = np.linspace(0, 5, 100)


    # ODE Int more oldschool, uses Fortran
    # solve_ivp more flexible

    # sol_m1 = odeint(dvdt, y0=v0, t=t, tfirst=True)
    # sol_m2 = solve_ivp(dvdt, t_span=(0, max(t)), y0=[v0], t_eval=t)

    # sol = odeint(dSdx, y0=S_0, t=x, tfirst=True)
    sol = solve_ivp(dSdt, t_span=(0, max(t)), y0=S_0, t_eval=t, method='DOP853', rtol=1e-10, atol=1e-13)

    x1_sol = sol.y[0]
    x2_sol = sol.y[1]
    t = sol.t

    # first parameter is velocity
    # print(sol_m1.T[0])

    print(sol)

    # v_sol_m1 = sol_m1.T[0]
    # v_sol_m2 = sol_m2.y[0]

    plt.plot(t, x1_sol)
    plt.plot(t, x2_sol)
    # plt.ylabel('$v(t)$', fontsize=22)
    # plt.xlabel('$t$', fontsize=22)
    plt.show()
