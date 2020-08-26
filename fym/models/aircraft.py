import gym
from gym import spaces

import numpy as np
import numpy.linalg as nla
from numpy import cos, sin
import scipy.linalg as sla
import scipy.interpolate
import scipy.optimize

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm, quat2angle, angle2quat


class Aircraft3Dof(BaseSystem):
    g = 9.80665
    rho = 1.2215
    m = 8.5
    S = 0.65
    b = 3.44
    CD0 = 0.033
    CD1 = 0.017
    name = 'aircraft'
    control_size = 2  # CL, phi
    state_lower_bound = [-np.inf, -np.inf, -np.inf, 3, -np.inf, -np.inf]
    state_upper_bound = [np.inf, np.inf, -0.01, np.inf, np.inf, np.inf]
    control_lower_bound = [-0.5, np.deg2rad(-70)]
    control_upper_bound = [1.5, np.deg2rad(70)]

    def __init__(self, initial_state, wind):
        super().__init__(self.name, initial_state, self.control_size)
        self.wind = wind
        self.term1 = self.rho*self.S/2/self.m

    def external(self, states, controls):
        state = states['aircraft']
        return dict(wind=self.wind.get(state))

    def deriv(self, state, t, control, external):
        CL, phi = control
        CD = self.CD0 + self.CD1*CL**2
        raw_control = CD, CL, phi
        return self._raw_deriv(state, t, raw_control, external)

    def _raw_deriv(self, state, t, control, external):
        x, y, z, V, gamma, psi = state
        CD, CL, phi = control
        (_, Wy, _), (_, (_, _, dWydz), _) = external['wind']

        dxdt = V*np.cos(gamma)*np.cos(psi)
        dydt = V*np.cos(gamma)*np.sin(psi) + Wy
        dzdt = - V*np.sin(gamma)

        dWydt = dWydz * dzdt

        dVdt = (-self.term1*V**2*CD - self.g*np.sin(gamma)
                - dWydt*np.cos(gamma)*np.sin(psi))
        dgammadt = (self.term1*V*CL*np.cos(phi) - self.g*np.cos(gamma)/V
                    + dWydt*np.sin(gamma)*np.sin(psi)/V)
        dpsidt = (self.term1*V/np.cos(gamma)*CL*np.sin(phi)
                  - dWydt*np.cos(psi)/V/np.cos(gamma))

        return np.array([dxdt, dydt, dzdt, dVdt, dgammadt, dpsidt])


class F16LinearLateral(BaseSystem):
    """
    Reference:
        B. L. Stevens et al. "Aircraft Control and Simulation", 3/e, 2016
        Example 5.3-1: LQR Design for F-16 Lateral Regulator
    Dynamics:
        x_dot = Ax + Bu
    State:
        x = [beta, phi, p, r, del_a, del_r, x_w]
        beta, phi: [rad], p, r: [rad/s], del_a, del_r: [deg]
    Control input:
        u = [u_a, u_r]  (aileron and rudder servo inputs, [deg])
    """
    A = np.array([
        [-0.322, 0.064, 0.0364, -0.9917, 0.0003, 0.0008, 0],
        [0, 0, 1, 0.0037, 0, 0, 0],
        [-30.6492, 0, -3.6784, 0.6646, -0.7333, 0.1315, 0],
        [8.5396, 0, -0.0254, -0.4764, -0.0319, -0.062, 0],
        [0, 0, 0, 0, -20.2, 0, 0],
        [0, 0, 0, 0, 0, -20.2, 0],
        [0, 0, 0, 57.2958, 0, 0, -1]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [20.2, 0],
        [0, 20.2],
        [0, 0]
    ])
    C = np.array([
        [0, 0, 0, 57.2958, 0, 0, -1],
        [0, 0, 57.2958, 0, 0, 0, 0],
        [57.2958, 0, 0, 0, 0, 0, 0],
        [0, 57.2958, 0, 0, 0, 0, 0]
    ])

    def __init__(self, initial_state=[1, 0, 0, 0, 0, 0, 0]):
        super().__init__(initial_state)

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u)


class MorphingPlane(BaseEnv):
    g = 9.80665  # [m/s^2]
    mass = 10  # [kg]
    S = 0.84  # reference area (norminal planform area) [m^2]
    # longitudinal reference length (nominal mean aerodynamic chord) [m]
    cbar = 0.288
    b = 3  # lateral reference length (nominal span) [m]
    Tmax = 50  # maximum thrust [N]

    control_limits = {
        "delt": (0, 1),
        "dele": np.deg2rad((-10, 10)),
        "dela": (-0.5, 0.5),
        "delr": (-0.5, 0.5),
        "eta1": (0, 1),
        "eta2": (0, 1),
    }

    coords = {
        "eta1": np.linspace(0, 1, 3),  # eta1
        "eta2": np.linspace(0, 1, 3),  # eta2
        "dele": np.deg2rad(np.linspace(-10, 10, 3)),  # dele
        "alpha": np.deg2rad(np.linspace(-10, 20, 61))  # alpha
    }

    polycoeffs = {
        "CD": [0.03802,
               [-0.0023543, 0.0113488, -0.00549877, 0.0437561],
               [[0.0012769, -0.00220993, 1166.938, 672.113],
                [0.00188837, 0.000115637, -203.85818, -149.4225],
                [-1166.928, 203.8535, 0.1956192, -115.13404],
                [-672.111624, 149.417, 115.76766, 0.994464]]],
        "CL": [0.12816,
               [0.13625538, 0.1110242, 1.148293, 6.0995634],
               [[-0.147822776, 1.064541, 243.35532, -330.0270179],
                [-1.13021511, -0.009309088, 166.28991, -146.8964467],
                [-243.282881, -166.2709286, 0.071258483, 4480.53564],
                [328.541707, 148.945785, -4480.67456545, -0.99765511]]],
        "Cm": [0.09406144,
               [-0.269902, 0.24346326, -7.46727, -2.7296],
               [[0.35794703, -7.433699, 647.83725, -141.0390569],
                [6.8532466, -0.0510021, 542.882121, -681.325],
                [-647.723162, -542.8638, 0.76322739, 2187.33517],
                [135.66547, 678.941, -2186.1196, 0.98880322]]]
    }

    J_template = np.array([
        [9323306930.82, -2622499.75, 56222833.68],
        [-2622499.75, 0, 395245.59],
        [56222833.68, 395245.59, 105244200037]
    ]) / 10**9 / 103.47649 * mass
    J_yy_data = np.array([
        [96180388451.54, 96468774320.55, 97352033548.31],
        [96180388451.54, 96720172843.10, 98272328292.52],
        [96180388451.54, 97077342563.70, 99566216309.81]
    ]).T / 10**9 / 103.47649 * mass
    J_yy = scipy.interpolate.interp2d(
        coords["eta1"], coords["eta2"], J_yy_data
    )

    def __init__(self, velocity, omega, quaternion, position):
        self.vel = BaseSystem(velocity, name="velocity")  # 3x1
        self.omega = BaseSystem(omega, name="omega")  # 3x1
        self.quat = BaseSystem(quaternion, name="quaternion")  # 4x1
        self.pos = BaseSystem(position, name="position")  # 3x1

    def J(self, eta1, eta2):
        J_temp = self.J_template
        J_temp[1, 1] = self.J_yy(eta1, eta2)
        return J_temp

    def _aero_base(self, name, *x):
        # x = [eta1, eta2, dele, alp]
        a0, a1, a2 = self.polycoeffs[name]
        return a0 + np.dot(a1, x) + np.sum(x * np.dot(a2, x), axis=0)

    def CD(self, eta1, eta2, dele, alp):
        return self._aero_base("CD", eta1, eta2, dele, alp)

    def CL(self, eta1, eta2, dele, alp):
        return self._aero_base("CL", eta1, eta2, dele, alp)

    def Cm(self, eta1, eta2, dele, alp):
        return self._aero_base("Cm", eta1, eta2, dele, alp)

    def set_dot(self, x, u, eta):
        v, omega, q, p = self.observe_list(x)
        F, M = self.aerodyn(v, q, p, u, eta)
        J = self.J(eta[0], eta[1])

        # force equation
        self.systems_dict["velocity"].dot = F / self.mass - np.cross(omega, v)

        # moment equation
        self.systems_dict["omega"].dot = (
            nla.inv(J).dot(M - np.cross(omega, J.dot(omega)))
        )

        # kinematic equation
        self.systems_dict["quaternion"].dot = 0.5 * np.append(
            -omega.dot(q[1:]),
            omega*q[0] - np.cross(omega, q[1:])
        )

        # navigation equation
        self.systems_dict["position"].dot = quat2dcm(q).T.dot(v)

    def state_readable(self, v=None, omega=None, q=None, p=None, preset="vel"):
        VT = sla.norm(v)
        alpha = np.arctan2(v[2], v[0])
        beta = np.arcsin(v[1] / VT)

        if preset == "vel":
            return VT, alpha, beta
        else:
            _, theta, _ = quat2angle(q)
            gamma = theta - alpha
            Q = omega[1]
            return {'VT': VT, 'gamma': gamma, 'alpha': alpha, 'Q': Q,
                    'theta': theta, 'beta': beta}

    def aerocoeff(self, *args):
        # *args: eta1, eta2, dele, alp
        # output: CL, CD, Cm, CC, Cl, Cn
        return self.CL(*args), self.CD(*args), self.Cm(*args), 0, 0, 0

    def aerodyn(self, v, q, p, u, eta):
        delt, dele, dela, delr = u
        x_cg, z_cg = 0, 0

        VT, alp, bet = self.state_readable(v=v, preset="vel")
        qbar = 0.5 * get_rho(-p[2]) * VT**2

        CL, CD, Cm, CC, Cl, Cn = self.aerocoeff(*eta, dele, alp)

        CX = cos(alp)*cos(bet)*(-CD) - cos(alp)*sin(bet)*(-CC) - sin(alp)*(-CL)
        CY = sin(bet)*(-CD) + cos(bet)*(-CC) + 0*(-CL)
        CZ = cos(bet)*sin(alp)*(-CD) - sin(alp)*sin(bet)*(-CC) + cos(alp)*(-CL)

        S, cbar, b, Tmax = self.S, self.cbar, self.b, self.Tmax

        X_A = qbar*CX*S  # aerodynamic force along body x-axis
        Y_A = qbar*CY*S  # aerodynamic force along body y-axis
        Z_A = qbar*CZ*S  # aerodynamic force along body z-axis

        # Aerodynamic moment
        l_A = qbar*S*b*Cl + z_cg*Y_A  # w.r.t. body x-axis
        m_A = qbar*S*cbar*Cm + x_cg*Z_A - z_cg*X_A  # w.r.t. body y-axis
        n_A = qbar*S*b*Cn - x_cg*Y_A  # w.r.t. body z-axis

        F_A = np.array([X_A, Y_A, Z_A])  # aerodynamic force [N]
        M_A = np.array([l_A, m_A, n_A])  # aerodynamic moment [N*m]

        # thruster force and moment are computed here
        T = Tmax*delt  # thrust [N]
        X_T, Y_T, Z_T = T, 0, 0  # thruster force body axes component [N]
        l_T, m_T, n_T = 0, 0, 0  # thruster moment body axes component [N*m]

        # Thruster force, momentum, and gravity force
        F_T = np.array([X_T, Y_T, Z_T])  # in body coordinate [N]
        M_T = np.array([l_T, m_T, n_T])  # in body coordinate [N*m]
        F_G = quat2dcm(q).dot(np.array([0, 0, self.mass*self.g]))

        F = F_A + F_T + F_G
        M = M_A + M_T

        return F, M

    def get_trim(self, z0={"alpha": 0.1, "delt": 0.13, "dele": 0},
                 fixed={"h": 300, "VT": 16, "eta": (0, 0)},
                 method="SLSQP", options={"disp": True, "ftol": 1e-10}):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            (self.coords["alpha"].min(), self.coords["alpha"].max()),
            self.control_limits["delt"],
            self.control_limits["dele"]
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        return self._trim_convert(result.x, fixed)

    def _trim_cost(self, z, fixed):
        x, u, eta = self._trim_convert(z, fixed)

        self.set_dot(x, u, eta)
        weight = np.diag([1, 1, 1000])

        dxs = np.append(self.vel.dot[(0, 2), ], self.omega.dot[1])
        return dxs.dot(weight).dot(dxs)

    def _trim_convert(self, z, fixed):
        h, VT, eta = fixed
        alp = z[0]
        v = np.array([VT*cos(alp), 0, VT*sin(alp)])
        omega = np.array([0, 0, 0])
        q = angle2quat(0, alp, 0)
        p = np.array([0, 0, -h])
        delt, dele, dela, delr = z[1], z[2], 0, 0

        x = np.hstack((v, omega, q, p))
        u = np.array([delt, dele, dela, delr])
        return x, u, eta


def get_rho(altitude):
    pressure = 101325 * (1 - 2.25569e-5 * altitude)**5.25616
    temperature = 288.14 - 0.00649 * altitude
    return pressure / (287*temperature)


class MorphingLon(BaseSystem, MorphingPlane):
    """
    This is a nonlinear simulator of morphing aircraft only considering
    the longitudinal dynamics for reduced computational burden.

    state (x)   : 4x1 vector of (V, alpha, q, gamma)
    control (u) : 2x1 vector of (delt, dele)
    morph (eta) : 2x1 vector of (eta1, eta2)
    """
    rho = get_rho(300)

    def __init__(self, init_state=None):
        if init_state is None:
            init_state, *_ = self.get_trim(verbose=True)

        super().__init__(init_state)

        self.limits = np.vstack([
            self.control_limits[k]
            for k in ("delt", "dele", "eta1", "eta2")
        ])[:, None].transpose(2, 0, 1)

    def deriv(self, x, u, eta):
        """Get the state derivative for given (x, u)"""
        self._check_control(u)

        V, alpha, q, gamma = x
        delt, dele = u
        eta1, eta2 = eta

        S, cbar, Tmax = self.S, self.cbar, self.Tmax
        m, g = self.mass, self.g

        qbar = self.rho * V**2 / 2
        T = Tmax * delt
        L = qbar * S * self.CL(eta1, eta2, dele, alpha)
        D = qbar * S * self.CD(eta1, eta2, dele, alpha)
        M = qbar * cbar * S * self.Cm(eta1, eta2, dele, alpha)
        Iy = self.J_yy(eta1, eta2)

        dV = 1 / m * (-D + T * cos(alpha) - m * g * sin(gamma))
        dalpha = q - 1 / (m * V) * (L + T * sin(alpha) - m * g * cos(gamma))
        dq = M / Iy
        dgamma = 1 / (m * V) * (L + T * sin(alpha) - m * g * cos(gamma))

        return np.vstack((dV, dalpha, dq, dgamma))

    def _check_control(self, u):
        tlim = self.control_limits["delt"]
        elim = self.control_limits["dele"]

        if tlim[0] > u[0] or tlim[1] < u[0]:
            print("WARN: thrust is over the limits")

        if elim[0] > u[1] or elim[1] < u[1]:
            print("WARN: elevator is over the limits")

    def _trim_cost(self, z, fixed):
        x, u, eta = self._trim_convert(z, fixed)
        dxs = self.deriv(x, u, eta)
        weight = np.diag([1, 1, 100, 1])
        return dxs.T.dot(weight).dot(dxs)[0][0]

    def _trim_convert(self, z, fixed):
        V, _, (eta1, eta2) = fixed
        alpha, delt, dele = z
        q, gamma = 0, 0

        x = np.vstack([V, alpha, q, gamma])
        u = np.vstack([delt, dele])
        eta = np.vstack([eta1, eta2])
        return x, u, eta

    def get_trim(self, z0={"alpha": 0.1, "delt": 0.19, "dele": 0},
                 fixed={"V": 20, "h": 300, "eta": (0.5, 0.5)},
                 method="SLSQP", options={"disp": False, "ftol": 1e-10},
                 verbose=False):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            (self.coords["alpha"].min(), self.coords["alpha"].max()),
            self.control_limits["delt"],
            self.control_limits["dele"]
        )
        result = scipy.optimize.minimize(
            self._trim_cost, z0, args=(fixed,),
            bounds=bounds, method=method, options=options)

        x, u, eta = self._trim_convert(result.x, fixed)
        dx = self.deriv(x, u, eta)

        if verbose:
            print("=========================================")
            print("               Trim point                ")
            print("               ----------                ")
            print("VT:   {:5.2f} [m/s]    AOA:   {:5.2f} [deg]".format(
                x[0, 0], x[1, 0] * np.rad2deg(1)))
            print("Q:    {:5.2f} [deg/s]  Gamma: {:5.2f} [deg]".format(
                x[2, 0] * np.rad2deg(1), x[3, 0] * np.rad2deg(1)))
            print("delt: {:5.2f} [ ]      dele:  {:5.2f} [deg]".format(
                u[0, 0], u[1, 0] * np.rad2deg(1)))
            print("eta1: {:5.2f} [ ]      eta2:  {:5.2f} [ ]".format(
                eta[0, 0], eta[1, 0]))

            print("")
            print("               Derivatives               ")
            print("               -----------               ")
            print("VT:  {:9.2e} [m/s]    AOA:   {:9.2e} [deg]".format(
                dx[0, 0], dx[1, 0] * np.rad2deg(1)))
            print("Q:   {:9.2e} [deg/s]  Gamma: {:9.2e} [deg]".format(
                dx[2, 0] * np.rad2deg(1), dx[3, 0] * np.rad2deg(1)))
            print("             (Norm: {:5.2e})".format(np.linalg.norm(dx)))
            print("=========================================")

        return x, u, eta

    def saturation(self, u):
        limits = np.vstack([
            self.control_limits[k]
            for k in ("delt", "dele", "eta1", "eta2")
        ])
        return np.clip(u, *limits.T)


class TransportLinearLongitudinal(BaseSystem):
    """
    Reference:
        B. L. Stevens et al. "Aircraft Control and Simulation", 3/e, 2016
        Example 4.6-4: Longitudinal Control for Automatic Landing
    """
    ap = np.array([
        [-0.038580, 18.984, -32.139, 0, 1.3233E-4, 0],
        [-0.0010280, -0.63253, 0.0056129, 1.0, 3.7553E-6, 0],
        [0, 0, 0, 1.0, 0, 0],
        [7.8601E-5, -0.75905, -0.00079341, -0.51830, -3.0808E-7, 0],
        [-0.043620, -249.76, 249.76, 0, 0, 0],
        [0, -250.00, 250.00, 0, 0, 0]
    ])
    bp = np.array([
        [10.100, 0],
        [-1.5446E-4, 0],
        [0, 0],
        [0.024656, -0.010770],
        [0, 0],
        [0, 0]
    ])
    cp = np.array([
        [0, 0, 57.2958, 0, 0, 0],
        [0, 0, 0, 57.2958, 0, 0, 0]
    ])

    def __init__(self, initial_state=[1, 0, 0, 0, 0, 0, 0]):
        super().__init__(initial_state)

    def deriv(self, x, u):
        return self.ap.dot(x) + self.bp.dot(u)


class TransportAugLinear(BaseSystem):
    """
    Reference:
        B. L. Stevens et al. "Aircraft Control and Simulation", 3/e, 2016
        Example 5.5-5: Glide-Slope Coupler
    """
    A = np.array([
        [-0.04, 19.0096, -32.1689, 0, 0, 10.1, 0, 0, 0, 0],
        [-0.001, -0.64627, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -0.7739, 0, -0.529765, 0, 0.02463, -0.011, 0, 0, 0],
        [0, -250, 250, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -10, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, -5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -10, 1],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0]
    ])
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, -4.3633],
        [0.2, 0, 0],
        [0, 10, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    C = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 57.2958, 0, 0, 0, 0, 0, 0],
        [0, 0, 57.2958, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0]
    ])
    G = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 1],
    ])

    def __init__(self, initial_state=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
        super().__init__(initial_state)

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u) + self.G.dot(np.array([0, 0]))
