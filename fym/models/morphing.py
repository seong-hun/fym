import fym
from fym.core import BaseSystem
import numpy as np
import numpy.linalg as nla
import math
import os

from numpy import sin, cos, arctan2, arcsin
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import LinearNDInterpolator
from scipy.io import loadmat


def get_rho(h):
    g = 9.80665
    R = 287.0531
    L = 0.0065
    hts = 11000
    rho0 = 1.225
    T0 = 288.15

    T = min(T0 - L*h, hts)
    expon = min(np.exp(g / (R * T) * (hts - h)), 1)
    rho = rho0 * (T / T0) ** (g / (L * R) - 1) * expon
    return rho


class MorphingAircraft(BaseSystem):
    def __init__(self, initial_state, name=None):
        super().__init__(initial_state, name)
        self.g = 9.80665
        self.mass = 10
        self.S = 0.84
        self.cbar = 0.288
        self.b = 3
        self.Tmax = 50

        self.dele_ll = (-10) * math.pi / 180
        self.dele_ul = (10) * math.pi / 180

        self.Tmax = 50
        self.zeta = 1
        self.omega_n = 20
        self.set_interpolators()

    def set_interpolators(self):
        self.eta1_grd = np.arange(-0.5, 1, 0.5)  # [-0.5, 0, 0.5]
        self.eta2_grd = np.arange(-0.5, 1, 0.5)
        self.alp_grd = np.arange(-10, 20.5, 0.5) * math.pi / 180
        self.dele_grd = [self.dele_ll, 0, self.dele_ul]
        eta_mesh = np.meshgrid(self.eta1_grd, self.eta2_grd)

        points = [
            (x, y) for x, y in zip(eta_mesh[0].ravel(), eta_mesh[1].ravel())
        ]

        x_cg = -(np.array([
            [183.22, 201.99, 216.61],
            [183.22, 206.27, 224.66],
            [183.22, 210.55, 232.71]
        ]) + 200 - 183.22) * 0
        x_cg = x_cg / 1000

        self.p1_interp = LinearNDInterpolator(points=points,  # ravel: 펼치기
                                              values=x_cg.ravel())

        Jyy_data = np.array([
            [96180388451.54, 96468774320.55, 97352033548.31],
            [96180388451.54, 96720172843.10, 98272328292.52],
            [96180388451.54, 97077342563.70, 99566216309.81]
        ])
        self.Jyy_interp = LinearNDInterpolator(
            points=points, values=Jyy_data.ravel())

        mesh = np.meshgrid(
            self.alp_grd, self.dele_grd, self.eta1_grd, self.eta2_grd)
        # points4 = [
        #     (u, v, w, z) for u, v, w, z in zip(mesh[0].ravel(),  #
        #                                        mesh[1].ravel(),
        #                                        mesh[2].ravel(),
        #                                        mesh[3].ravel())
        # ]
        points4 = [
            (u, v, w, z) for u, v, w, z in zip(*map(np.ravel, mesh))
        ]

        mat = loadmat(os.path.join(fym.__path__[0], "models", "aero_coeff.mat"))
        CD = mat['CD']
        CL = mat['CL']
        Cm = mat['CM']

        CF_S = np.array([
            [0.840, 0.820, 0.810],
            [0.984, 0.962, 0.949],
            [1.129, 1.104, 1.087]
        ])
        CF_cbar = np.array([
            [0.288, 0.299, 0.351],
            [0.275, 0.286, 0.336],
            [0.265, 0.276, 0.325]
        ])
        # CF_b = np.array([
        #     [3.000, 2.810, 2.354],
        #     [3.722, 3.490, 2.908],
        #     [4.446, 4.170, 3.462]
        # ])
        CD_f = 0.4
        S_f = 0.084

        CD_grd = CD.transpose(3, 2, 0, 1)
        CL_grd = CL.transpose(3, 2, 0, 1) + CD_f * S_f / CF_S[0, 0]
        Cm_grd = Cm.transpose(3, 2, 0, 1)

        CD_grd = CD_grd / CF_S[0, 0] * CF_S
        CL_grd = CL_grd / CF_S[0, 0] * CF_S
        Cm_grd = Cm_grd / CF_S[0, 0] * CF_S / CF_cbar[0, 0] * CF_cbar

        # for i in range(0, 3):
        #     for j in range(0, 3):
        #         CD_grd[:, :, i, j] = (
        #             CD_grd[:, :, i, j] / CF_S[0, 0] * CF_S[i, j])
        #         CL_grd[:, :, i, j] = (
        #             CL_grd[:, :, i, j] / CF_S[0, 0] * CF_S[i, j])
        #         Cm_grd[:, :, i, j] = (
        #             Cm_grd[:, :, i, j] / CF_S[0, 0] * CF_S[i, j]
        #             / CF_cbar[0, 0] * CF_cbar[i, j])

        self.CD_interp = LinearNDInterpolator(points=points4,
                                              values=CD_grd.ravel())

        self.CL_interp = LinearNDInterpolator(points=points4,
                                              values=CL_grd.ravel())

        self.Cm_interp = LinearNDInterpolator(points=points4,
                                              values=Cm_grd.ravel())

    def get_J(self, eta1, eta2):
        # eta1, eta2 = np.clip(
        #     eta,
        #     [self.eta1_grd.min(), self.eta2_grd.min()],
        #     [self.eta1_grd.max(), self.eta2_grd.max()]
        # )  # controller 짤 때 clip
        J_xx = 9323306930.82
        J_zz = 105244200037
        J_xy = -2622499.75
        J_xz = 56222833.68
        J_yz = 395245.59

        J = np.array([
            [J_xx, J_xy, J_xz],
            [J_xy, self.Jyy_interp(eta1, eta2), J_yz],
            [J_xz, J_yz, J_zz]
        ]) * self.mass / 1e9 / 103.47649
        return J

    def get_p_ref(self, eta1, eta2):
        p_ref = np.array([self.p1_interp(eta1, eta2), 0, 0])
        return p_ref

    def aero_model(self, alp, bet, VT, h, eta, P, Q, R, dela, dele, delr):
        eta1 = eta[0]
        eta2 = eta[1]

        CD = self.CD_interp(alp, dele, eta1, eta2)
        CL = self.CL_interp(alp, dele, eta1, eta2)
        Cm = self.Cm_interp(alp, dele, eta1, eta2)

        CC = Cl = Cn = 0

        CX = (
            cos(alp) * cos(bet) * (-CD) - cos(alp) * sin(bet) * (-CC)
            - sin(alp) * (-CL))
        CY = (
            sin(bet) * (-CD) + cos(bet) * (-CC) + 0 * (-CL))
        CZ = (
            cos(bet) * sin(alp) * (-CD) - sin(alp) * sin(bet) * (-CC)
            + cos(alp) * (-CL))

        aero_coeff = np.array([CX, CY, CZ, Cl, Cm, Cn])
        return aero_coeff

    def aerodyn(self, x, u):
        S, cbar, b, Tmax = self.S, self.cbar, self.b, self.Tmax

        v_w = [0, 0, 0]
        delt = u[0]
        dela = u[1]
        dele = u[2]
        delr = u[3]
        eta = u[4:6]

        p_cg = self.get_p_ref(*eta)
        x_cg = p_cg[0]
        z_cg = p_cg[2]

        state13 = x
        # U = state13[0]
        # V = state13[1]
        # W = state13[2]
        P = state13[3]
        Q = state13[4]
        R = state13[5]
        # q_0 = state13[6]
        # q_1 = state13[7]
        # q_2 = state13[8]
        # q_3 = state13[9]
        # p_N = state13[10]
        # p_E = state13[11]
        p_D = state13[12]
        h = -p_D

        v = state13[:3]
        q = state13[6:10]

        v_rel = v - rot.from_quat(q).as_dcm().T.dot(v_w)
        U_rel, V_rel, W_rel = v_rel

        VT = nla.norm(v_rel)
        alp = arctan2(W_rel, U_rel)
        bet = arcsin(V_rel / VT)

        rho = get_rho(h)
        qbar = 0.5 * rho * VT ** 2

        CX, CY, CZ, Cl, Cm, Cn = self.aero_model(
            alp, bet, VT, h, eta, P, Q, R, dela, dele, delr)

        X_A = qbar * CX * S
        Y_A = qbar * CY * S
        Z_A = qbar * CZ * S

        l_A = qbar * S * b * Cl + z_cg * Y_A
        m_A = qbar * S * cbar * Cm + x_cg * Z_A - z_cg * X_A
        n_A = qbar * S * b * Cn - x_cg * Y_A

        F_A = np.array([X_A, Y_A, Z_A])
        M_A = np.array([l_A, m_A, n_A])

        T = Tmax * delt
        X_T = T
        Y_T = Z_T = l_T = m_T = n_T = 0

        F_T = np.array([X_T, Y_T, Z_T])
        M_T = np.array([l_T, m_T, n_T])

        return F_A + F_T, M_A + M_T

    def deriv(self, x, u):
        F, M = self.aerodyn(x, u)
        eta = u[4:6]

        J = self.get_J(*eta)

        v = x[0:3]
        omega = x[3:6]
        q = x[6:10]

        dcm = rot.from_quat(q).as_dcm()
        F_g = dcm.dot([0, 0, self.mass * self.g])
        M_g = np.zeros(3)

        F = F + F_g
        M = M + M_g

        dv = F / self.mass - np.cross(omega, v)
        domega = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega)))
        dq = 0.5*np.hstack([
            -np.dot(omega, q[1:4]),
            omega*q[0] - np.cross(omega, q[1:4])
        ])
        dp = dcm.T.dot(v)

        xdot = np.hstack([dv, domega, dq, dp])  # 기본적으로 row vector로 구성됨
        return xdot


if __name__ == '__main__':
    system = MorphingAircraft(name='morphingaircraft',
                              initial_state=[0, 0, 0, 0])
