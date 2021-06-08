import fym.utils.parser as parser
import agent


parser.update(agent.cfg.multicopter, {
    "LQRGain": {
        "Q": [2, 2, 2, 2],
        "R": [2, 2]
    }
})

lqr = agent.LQR()
print(lqr.Q, lqr.R)
