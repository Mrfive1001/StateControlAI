from SmallStateControl import SSCPENV
import A3C

env = SSCPENV()
para = A3C.Para(env,
                MAX_GLOBAL_EP=2000,
                UPDATE_GLOBAL_ITER=50,
                GAMMA=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001,)
RL = A3C.A3C(para)
RL.run()
