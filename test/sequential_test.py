import fym


class Link(fym.BaseEnv):
    name = "link"

    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(shape=(3, 1))


class Env(fym.BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=10)
        self.links1 = fym.Sequential(**{f"link{i:02d}": Link() for i in range(5)})
        self.links2 = fym.Sequential(*(Link() for i in range(5)))


env = Env()
