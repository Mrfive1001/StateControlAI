import numpy as np
from SmallStateControl import SSCPENV as SSP

# 连雪、、

class SSCPENV(SSP):
    def __init__(self, x_dim=2, action_dim=1, init_x=None):
        SSP.__init__(x_dim=2, action_dim=1, init_x=None)
        self.abound = np.linspace(0,12,5)
        self.n_action = len(self.abound)

    def step(self, omega):
        if self.action_dim == 1:
            if type(omega) == np.ndarray:
                omega = omega[0]
        omega = self.abound[int(omega)]
        return SSP.step(omega)