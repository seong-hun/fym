'''
PID control for SISO (single input single output) system
'''
import numpy as np


class PID():
    def __init__(self):
        self.dt = 0.01
        self.integration = Integration_Euler()
        self.differentiation = Differentiation_Euler()

    def input(self, y: float, y_ref: float, gain: np.array) -> float:
        if len(gain) != 3:
            print("""PID gain must consist of three elements,
                  i.e., p, i, and d.""")
        else:
            p = gain[0]
            i = gain[1]
            d = gain[2]

        e = y - y_ref
        e_i = self.integration.integ(self.integration.e_i, e, self.dt)
        e_d = self.differentiation.deriv(
            self.differentiation.e_prev, e, self.dt)
        u = p * e + i * e_i + d * e_d
        return u


class Integration():
    def __init__(self):
        self.anti_windup = 0            # 1: anti_windup
        self.anti_windup_min = -100
        self.anti_windup_max = 100
        self.e_i = []

    def integ(self):
        raise NotImplementedError

    def windup(self, x: float, x_min: float, x_max: float) -> float:
        if self.anti_windup == 0:
            x_windup = x
        else:
            if x > x_max:
                x_windup = x_max
            elif x < x_min:
                x_windup = x_min
            else:
                x_windup = x
                return x_windup


class Integration_Euler(Integration):
    def integ(self, e_i: float, e: float, dt: float) -> float:
        if not self.e_i:
            self.e_i = 0
        else:
            e_i = e_i + e * dt
            self.e_i = self.windup(e_i,
                                   self.anti_windup_min, self.anti_windup_max)
        return self.e_i


class Differentiation():
    def __init__(self):
        self.e_prev = 0         # initial guess for numerical differentiation

    def deriv(self):
        raise NotImplementedError


class Differentiation_Euler(Differentiation):
    def deriv(self, e_prev: float, e: float, dt: float) -> float:
        self.e_d = (e - e_prev) / dt
        self.e_prev = e
        return self.e_d


"""
below codes: test
"""
y = 1
y_ref = 2
gain = np.array([1, 2, 3])
ctrllr = PID()
print(ctrllr.input(y, y_ref, gain))
print(ctrllr.integration.e_i,
      ctrllr.differentiation.e_d, ctrllr.differentiation.e_prev)
