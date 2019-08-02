import gym
from gym import spaces
import numpy as np
from nrfsim.core import BaseSystem


class ElectricWheelchair(BaseSystem):
    g = 9.80665
    mWheelchair = 65.9      # mass of wheelchair' mass
    Jb = np.identity(3)       # moment of inertia
    Rrear = 0.5    # rear wheel' radius
    width = 0.8
    depth = 1
    height = 1
    # vector from body origin th right caster
    rCRBb = np.array([depth/2, width/2, height/2])
    # vector from body origin th left4 caster
    rCLBb = np.array([depth/2, -width/2, height/2])
    # vector from body origin th right caster
    rWRBb = np.array([-depth/2, width/2, height/2])
    # vector from body origin th left4 caster
    rWLBb = np.array([-depth/2, -width/2, height/2])
    Kalpha = 0.01    # caster angle's gain
    kr = 0.1    # rear wheel's friction coefficient
    kch = 0.1   # caster's horizontal direction friction coefficient
    kcv = 0.1   # caster's vertical direction friction coefficient
    name = 'electricWheelchair'
    state_lower_bound = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, -np.inf, -np.inf, -np.pi,
                        -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 
                        -np.pi, -np.pi],
    state_upper_bound = [np.inf, np.inf, np.inf, np.inf, np.inf,
                        np.inf, np.inf, np.inf, np.inf, np.pi,
                        np.pi, np.pi, np.inf, np.inf, np.inf, 
                        np.pi, np.pi]
    control_lower_bound = [-1.87, -1.87]  # Left, Right wheel torque
    control_upper_bound = [1.87, 1.87]
    control_size = np.array(control_lower_bound).size

    def __init__(self, initial_state, mload, rGBb):
        super().__init__(self.name, initial_state, self.control_size)
        self.mload = mload
        self.rGBb = rGBb

    def external(self, states, controls):
        state = states['electricWheelchair']
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, alphaL, alphaR = state
        CRcb = np.array([[np.cos(alphaR), np.sin(alphaR), 0],
                [-np.sin(alphaR), np.cos(alphaR), 0],
                [0, 0, 1]])       # coordinate transform matrix from body to right caster
        CLcb = np.array([[np.cos(alphaL), np.sin(alphaL), 0],
                [-np.sin(alphaL), np.cos(alphaL), 0],
                [0, 0, 1]])       # coordinate transform matrix from body to left caster
        CRbc = np.transpose(CRcb)
        CLbc = np.transpose(CLcb)
        Cx = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [
            0, -np.sin(phi), np.cos(phi)]])
        Cy = np.array([[np.cos(theta), 0, -np.sin(theta)],
              [0, 1, 0],
              [np.sin(theta), 0, np.cos(theta)]])
        Cz = np.array([[np.cos(psi), np.sin(psi), 0],
              [-np.sin(psi), np.cos(psi), 0],
              [0, 0, 1]])
        Cbi = Cx*Cy*Cz
        Cib = np.transpose(Cbi)
        m = mWheelchair + self.mload
        Fgi = np.array([0, 0, m*g])
        Fgb = Cbi*Fgi
        angVelb = np.array([p, q, r])
        # velocity at left caster's position in body coordinate
        velCLb = Cbi*veli + np.cross(angVelb, rCLBb)
        # velocity at right caster's position in body coordinate
        velCRb = Cbi*veli + np.coss(angVelb, rCRBb)
        # velocity at lefr rear wheel's position in body coordinate
        velWLb = Cbi*veli + np.cross(angVelb, rWLBb)
        # velocity at right rear wheel's position in body coordinate
        velWRb = Cbi*veli + np.coss(angVelb, rWRBb)
        velCLc = CLcb*velCLb
        velCRc = CRcb*velCRb

        n1 = Fgb[2]*(1-2*self.rGB[1]/width)*(1-2*self.rGB[0]/depth) / \
                     4       # normal force at left rear wheel
        n2 = Fgb[2]*(1+2*self.rGB[1]/width)*(1-2*self.rGB[0]/depth) / \
                     4       # normal force at right rear wheel
        n3 = Fgb[2]*(1-2*self.rGB[1]/width)*(1+2*self.rGB[0] /
                     depth)/4       # normal force at left caster
        n4 = Fgb[2]*(1+2*self.rGB[1]/width)*(1+2*self.rGB[0] /
                     depth)/4       # normal force at right caster

        # friction at left rear wheel in body coordinate
        F3b = np.array([-np.sign(velLb[0])*kr*n1, 0, 0])
        # friction at right rear wheel in body coordinate
        F4b = np.array([-np.sign(velRb[0])*kr*n2, 0, 0])

        # horizontal friction at left caster in caster coordinate
        FCLhc = np.array([-np.sign(velCRL[0])*kch*n3, 0, 0])
        # horizontal friction at right caster in caster coordinate
        FCRhc=np.array([-np.sign(velCRc[0])*kch*n4, 0, 0])
        # vertical friction at left caster in caster coordinate
        FCLvc=np.array([0, -np.sign(velCLc[1])*kcv*n3, 0])
        # vertical friction at right caster in caster coordinate
        FCRvc=np.array([0, -np.sign(velCRc[1])*kcv*n4, 0])

        FCRhb=CRbc*FCRhc
        FCLhb=CRbc*FCLhc
        FCRvb=CRbc*FCRvc
        FCLvb=CRbc*FCLvc

        # x-direction reaction force from left caster to body
        f5b=FCLhb[0] + FCLvb[0]
        # x-direction reaction force from right caster to body
        f6b=FCRhb[0] + FCRvb[0]
        # y-direction reaction force from left caster to body
        f7b=FCLhb[1] + FCLvb[1]
        # y-direction reaction force from right caster to body
        f8b=FCRhb[1] + FCRvb[1]

        F5b=np.array([f5b, 0, 0])
        F6b=np.array([f6b, 0, 0])
        F7b=np.array([0, f7b, 0])
        F8b=np.array([0, f8b, 0])

        N1b=np.array([0, 0, -n1])
        N2b=np.array([0, 0, -n2])
        N3b=np.array([0, 0, -n3])
        N4b=np.array([0, 0, -n4])

        return [F3b, F4b, F5b, F6b, F7b, F8b, N1b, N2b, N3b, N4b, FCLvc, FCRvc]

    def deriv(self, state, t, control, external):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, alphaL, alphaR=state
        F3b, F4b, F5b, F6b, F7b, F8b, N1b, N2b, N3b, N4b, FCLvc, FCRvc=external
        Cx=np.array([[1, 0, 0],
              [0, np.cos(phi), np.sin(phi)],
              [0, - np.sin(phi), np.cos(phi)]])
        Cy=np.array([[np.cos(theta), 0, -np.sin(theta)],
              [0, 1, 0],
              [np.sin(theta), 0, np.cos(theta)]])
        Cz=np.array([[np.cos(psi), np.sin(psi), 0],
              [-np.sin(psi), np.cos(psi), 0],
              [0, 0, 1]])
        Cbi=Cx*Cy*Cz
        Cib=np.transpose(Cbi)
        m=mWheelchair + self.mload
        Fgi=np.array([0, 0, m*g])
        angVelb=np.array([p, q, r])
        T1, T2=control
        F1i=Cib*np.array([T1/Rrear, 0, 0])
        F2i=Cib*np.array([T2/Rrear, 0, 0])
        F3i=Cib*F3b
        F4i=Cib*F4b
        F5i=Cib*F5b
        F6i=Cib*F6b
        F7i=Cib*F7b
        F8i=Cib*F8b

        Ftotali=F1i + F2i + F3i + F4i + F5i + F6i + F7i + \
            F8i + Fgi + Cib*N1b + Cib*N2b + Cib*N3b + Cib*N4b

        Mb=(np.cross(rWLBb, Cbi*F1i) + np.cross(rWRBb, Cbi*F2i) + np.cross(rWLBb, Cbi*F3i) + np.cross(rWRBb, Cbi*F4i) + np.cross(rCLBb, Cbi*F5i) +
              np.cross(rCRBb, Cbi*F6i) + np.cross(rCLBb, Cbi*F7i) + np.cross(rCRBb, Cbi*F8i) + np.cross(rWLBb, N1b) + np.cross(rWRBb, N2b) +
              np.cross(rCLBb, N3b) + np.cross(rCRBb, N4b) + np.cross(self.rGB, Cbi*Fgi))
        
        euldot=np.array([[1, np.tan(theta)*np.sin(phi), np.tan(theta)*np.cos(phi)],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])*angVelb
        angAccb=np.linalg.inv(Jb)*(Mb - np.cross(angVelb, Jb*angVelb))
        veli=np.array([vx, vy, vz])
        acci=Ftotali/m - np.cross(Cib*angAccb, Cib*self.rGBb) - \
            np.cross(Cib*angVelb, np.cross(Cib*angVelb, Cib*self.rGBb))

        '''
        velCLb = Cbi*veli + np.cross(angVelb, rCLBb)
        velCRb = Cbi*veli + np.coss(angVelb, rCRBb)
        alphaLtarget = np.arctan2(velCLb[1], velCLb[0])
        alphaRtarget = np.arctan2(velCRb[1], velCRb[0])
        alphaLdot = -np.sign(FCLvc[1])*Kalpha*(alphaLtarget - alphaL)
        alphaRdot = -np.sign(FCRvc[1])*Kalpha*(alphaRtarget - alphaR)
        '''
        alphaLdot=-np.sign(FCLvc[1])*Kalpha
        alphaRdot=-np.sign(FCRvc[1])*Kalpha

        alphadot=np.array([alphaLdot, alphaRdot])

        return np.hstack([veli, acci, euldot, angAccb, alphadot])
