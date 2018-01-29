from SmallStateControl import SSCPENV
import A3C

env = SSCPENV()
para = A3C.Para(env,
                units_a=30,
                units_c=100,
                MAX_GLOBAL_EP=2000,
                UPDATE_GLOBAL_ITER=30,
                gamma=0.9,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.001, )
RL = A3C.A3C(para)
RL.run()
