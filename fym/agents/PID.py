'''
PID control for SISO (single input single output) system
'''
import numpy as np


class PID():
    def __init__(self, gain, windup=False, method='euler', dt=0.01):
        self.e_intg = 0
        self.e_prev = 0  # initial guess for differentiator
        self.windup = windup
        self.dt = dt
        if len(gain) != 3:
            print("PID gain must consist of three elements,"
                  "i.e., p, i, and d.")
        else:
            self.p = gain[0]
            self.i = gain[1]
            self.d = gain[2]

        if method == 'euler':
            self.integrate = intg_euler
            self.differentiate = diff_euler

    def get(self, e: float) -> float:
        dt = self.dt
        e_i = self.e_intg
        e_d = self.differentiate(e, self.e_prev, dt)
        u = self.p * e + self.i * e_i + self.d * e_d

        # Update
        if self.windup:
            self.e_intg = self.int_windup(self.integrate(e, self.e_intg, dt))
        else:
            self.e_intg = self.integrate(e, self.e_intg, dt)
        self.e_prev = e

        return u

    def int_windup(self, x: float) -> float:
        x_max = self.windup
        x_min = -self.windup
        if x > x_max:
            x = x_max
        elif x < x_min:
            x = x_min
        return x


def intg_euler(e, e_intg, dt):
    return e_intg + e * dt


def diff_euler(e, e_prev, dt):
    return (e - e_prev) / dt


if __name__ == '__main__':
    e = 100
    gain = np.array([1, 2, 3])
    ctrllr = PID(gain)
    print(ctrllr.input(e))
    print(ctrllr.e_intg, ctrllr.e_prev)
    e = -100
    gain = np.array([1, 3, 3])
    print(ctrllr.input(e))
    print(ctrllr.e_intg, ctrllr.e_prev)
