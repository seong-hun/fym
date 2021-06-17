import fym
import fym.core


class Link(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(shape=(3, 1))


class Env(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        self.links = fym.core.Sequential(
            **{f"link{i:02d}": Link() for i in range(5)}
        )


env = Env()
