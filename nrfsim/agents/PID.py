'''
PID control for SISO (single input single output) system
'''
import numpy as np


class PID():
    def __init__(self, method='euler', dt=0.01, windup=True):
        self.e_intg = 0
        self.e_prev = 0  # initial guess for differentiator
        self.windup = windup
        self.dt = dt

        if method == 'euler':
            self.integrate = intg_euler
            self.differentiate = diff_euler

    def input(self, e: float, gain: np.ndarray) -> float:
        if len(gain) != 3:
            print("PID gain must consist of three elements,"
                  "i.e., p, i, and d.")
        else:
            p = gain[0]
            i = gain[1]
            d = gain[2]

        dt = self.dt
        e_i = self.e_intg
        e_d = self.differentiate(e, self.e_prev, dt)
        u = p * e + i * e_i + d * e_d

        # Update
        if self.windup:
            self.e_intg = windup(self.integrate(e, self.e_intg, dt))
        else:
            self.e_intg = self.integrate(e, self.t_state, dt)
        self.e_prev = e

        return u


def windup(x: float, x_min=-100, x_max=100) -> float:
    if x > x_max:
        x = x_max
    elif x < x_min:
        x = x_min
    return x


def intg_euler(e, e_intg, dt):
    return e_intg + e * dt


def diff_euler(e, e_prev, dt):
    return (e - e_prev) / dt


"""
below codes: test
"""
y = 1
y_ref = 2
e = y - y_ref
gain = np.array([1, 2, 3])
ctrllr = PID()
print(ctrllr.input(e, gain))
# print(ctrllr.e_intg, ctrllr.e_prev)
