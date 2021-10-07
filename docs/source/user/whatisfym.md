# What is Fym?

**Fym** is a general perpose dynamical simulator based on
[Python](https://www.python.org). The origin of **Fym** is a flight simulator
that requires highly accurate integration (e.g. [Runge-Kutta-Fehlberg
method](https://en.wikipedia.org/wiki/Runge–Kutta–Fehlberg_method) or simply
`rk45`), and a set of components that interact each other. For the integration,
**Fym** supports various [Scipy
integration](https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html)
methods in addition with own fixed-step solver such as `rk4`. Also, **Fym** has
a novel structure that provides modular design of each component of systems,
which is much simiar to
[Simulink](https://kr.mathworks.com/products/simulink.html).

The **Fym** project began with the development of accurate flight simulators
that aerospace engineers could use with [OpenAI Gym](https://gym.openai.com) to
study reinforcement learning. This is why the package name is **Fym** (Flight +
Gym). Although it is now a general purpose dynamical simulator, many codes and
ideas have been devised in the OpenAI Gym.
