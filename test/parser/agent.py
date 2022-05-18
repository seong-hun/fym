import fym.utils.parser as parser

cfg = parser.parse(
    {
        "env.kwargs": dict(dt=0.01, max_t=10),
        "multicopter.nrotor": 6,
        "multicopter.m": 3.0,
        "multicopter.LQRGain": {
            "Q": [1, 1, 1, 1],
            "R": [1, 1],
        },
        "actuator.tau": 1e-1,
    }
)


class LQR:
    def __init__(self):
        self.Q = cfg.multicopter.LQRGain.Q
        self.R = cfg.multicopter.LQRGain.R
