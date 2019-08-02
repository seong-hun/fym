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
                        -np.inf, -np.pi,
                        -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 
                        -np.pi, -np.pi],
    state_upper_bound = [np.inf, np.inf, np.inf, np.inf, np.inf,
                        np.inf, np.pi,
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
        Cbi = Cx.dot(Cy).dot(Cz)
        Cib = np.transpose(Cbi)
        m = self.mWheelchair + self.mload
        Fgi = np.array([0, 0, m*self.g])
        Fgb = Cbi.dot(Fgi)
        angVelb = np.array([p, q, r])
        veli=np.array([vx, vy, vz])
        # velocity at left caster's position in body coordinate
        velCLb = Cbi.dot(veli) + np.cross(angVelb, self.rCLBb)
        # velocity at right caster's position in body coordinate
        velCRb = Cbi.dot(veli) + np.cross(angVelb, self.rCRBb)
        # velocity at lefr rear wheel's position in body coordinate
        velWLb = Cbi.dot(veli) + np.cross(angVelb, self.rWLBb)
        # velocity at right rear wheel's position in body coordinate
        velWRb = Cbi.dot(veli) + np.cross(angVelb, self.rWRBb)
        velCLc = CLcb.dot(velCLb)
        velCRc = CRcb.dot(velCRb)

        n1 = Fgb[2]*(1-2*self.rGBb[1]/self.width)*(1-2*self.rGBb[0]/self.depth) / \
                     4       # normal force at left rear wheel
        n2 = Fgb[2]*(1+2*self.rGBb[1]/self.width)*(1-2*self.rGBb[0]/self.depth) / \
                     4       # normal force at right rear wheel
        n3 = Fgb[2]*(1-2*self.rGBb[1]/self.width)*(1+2*self.rGBb[0] /
                     self.depth)/4       # normal force at left caster
        n4 = Fgb[2]*(1+2*self.rGBb[1]/self.width)*(1+2*self.rGBb[0] /
                     self.depth)/4       # normal force at right caster

        # friction at left rear wheel in body coordinate
        F3b = np.array([-np.sign(velWLb[0])*self.kr*n1, 0, 0])
        # friction at right rear wheel in body coordinate
        F4b = np.array([-np.sign(velWRb[0])*self.kr*n2, 0, 0])

        # horizontal friction at left caster in caster coordinate
        FCLhc = np.array([-np.sign(velCLc[0])*self.kch*n3, 0, 0])
        # horizontal friction at right caster in caster coordinate
        FCRhc=np.array([-np.sign(velCRc[0])*self.kch*n4, 0, 0])
        # vertical friction at left caster in caster coordinate
        FCLvc=np.array([0, -np.sign(velCLc[1])*self.kcv*n3, 0])
        # vertical friction at right caster in caster coordinate
        FCRvc=np.array([0, -np.sign(velCRc[1])*self.kcv*n4, 0])

        FCRhb=CRbc.dot(FCRhc)
        FCLhb=CRbc.dot(FCLhc)
        FCRvb=CRbc.dot(FCRvc)
        FCLvb=CRbc.dot(FCLvc)

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
        Cbi=Cx.dot(Cy).dot(Cz)
        Cib=np.transpose(Cbi)
        m=self.mWheelchair + self.mload
        Fgi=np.array([0, 0, m*self.g])
        angVelb=np.array([p, q, r])
        T1, T2=control
        F1i=Cib.dot(np.array([T1/self.Rrear, 0, 0]))
        F2i=Cib.dot(np.array([T2/self.Rrear, 0, 0]))
        #import ipdb; ipdb.set_trace()
        F3i=Cib.dot(F3b)
        F4i=Cib.dot(F4b)
        F5i=Cib.dot(F5b)
        F6i=Cib.dot(F6b)
        F7i=Cib.dot(F7b)
        F8i=Cib.dot(F8b)

        Ftotali=F1i + F2i + F3i + F4i + F5i + F6i + F7i + \
            F8i + Fgi + Cib.dot(N1b) + Cib.dot(N2b) + Cib.dot(N3b) + Cib.dot(N4b)
        #import ipdb; ipdb.set_trace()
        Mb=(np.cross(self.rWLBb, Cbi.dot(F1i)) + np.cross(self.rWRBb, Cbi.dot(F2i)) + np.cross(self.rWLBb, Cbi.dot(F3i)) + np.cross(self.rWRBb, Cbi.dot(F4i)) + np.cross(self.rCLBb, Cbi.dot(F5i)) +
              np.cross(self.rCRBb, Cbi.dot(F6i)) + np.cross(self.rCLBb, Cbi.dot(F7i)) + np.cross(self.rCRBb, Cbi.dot(F8i)) + np.cross(self.rWLBb, N1b) + np.cross(self.rWRBb, N2b) +
              np.cross(self.rCLBb, N3b) + np.cross(self.rCRBb, N4b) + np.cross(self.rGBb, Cbi.dot(Fgi)))
        
        euldot=np.array([[1, np.tan(theta)*np.sin(phi), np.tan(theta)*np.cos(phi)],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]]).dot(angVelb)
        angAccb=np.linalg.inv(self.Jb).dot((Mb - np.cross(angVelb, self.Jb.dot(angVelb))))
        veli=np.array([vx, vy, vz])
        acci=Ftotali/m - np.cross(Cib.dot(angAccb), Cib.dot(self.rGBb)) - \
            np.cross(Cib.dot(angVelb), np.cross(Cib.dot(angVelb), Cib.dot(self.rGBb)))

        '''
        velCLb = Cbi*veli + np.cross(angVelb, rCLBb)
        velCRb = Cbi*veli + np.coss(angVelb, rCRBb)
        alphaLtarget = np.arctan2(velCLb[1], velCLb[0])
        alphaRtarget = np.arctan2(velCRb[1], velCRb[0])
        alphaLdot = -np.sign(FCLvc[1])*Kalpha*(alphaLtarget - alphaL)
        alphaRdot = -np.sign(FCRvc[1])*Kalpha*(alphaRtarget - alphaR)
        '''
        alphaLdot=-np.sign(FCLvc[1])*self.Kalpha
        alphaRdot=-np.sign(FCRvc[1])*self.Kalpha

        alphadot=np.array([alphaLdot, alphaRdot])

        return np.hstack([veli, acci, euldot, angAccb, alphadot])
