import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# following tutorial
# https://www.youtube.com/watch?v=MM3cBamj1Ms


def dvdt(t, v):
    return 3*v**2 - 5

v0 = 0


# coupled first_order ODE
#y1' = y1 + y2^2 + 3x
#y2' = 3y1 + y2^3 - cos(x)
# y1(0) = (0)
# y2(0) = 0
# x is our independent variable (kinda like t)

# S = (y1, y2)
# dS/dx = (dy1/dx, dy2/dx)
# just want our vector from its derivative!

def dSdx(x, S):
    y1, y2 = S
    return [y1 + y2**2 + 3*x,
            3*y1 + y2**3 - np.cos(x)]
y1_0 = 0
y2_0 = 0
S_0 = (y1_0, y2_0)





if __name__ == "__main__":
    # setup what x will be!
    x = np.linspace(0, 1, 100)

    # sol_m1 = odeint(dvdt, y0=v0, t=t, tfirst=True)
    # sol_m2 = solve_ivp(dvdt, t_span=(0, max(t)), y0=[v0], t_eval=t)

    sol = odeint(dSdx, y0=S_0, t=x, tfirst=True)

    y1_sol = sol.T[0]
    y2_sol = sol.T[1]

    # first parameter is velocity
    # print(sol_m1.T[0])

    print(sol)

    # v_sol_m1 = sol_m1.T[0]
    # v_sol_m2 = sol_m2.y[0]

    plt.plot(x, y1_sol)
    plt.plot(x, y2_sol)
    # plt.ylabel('$v(t)$', fontsize=22)
    # plt.xlabel('$t$', fontsize=22)
    plt.show()