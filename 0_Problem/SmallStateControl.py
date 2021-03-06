import numpy as np


class SSCPENV(object):
    def __init__(self, x_dim=2, action_dim=1, init_x=None):
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.abound = np.array([0, 20])
        self.init_x = init_x
        self.state_dim = self.x_dim
        self.t = 0
        self.delta_t = 0.01
        self.xd = 5
        self.xd_dot = 0
        self.xd_dot2 = 0
        self.total_time = 5
        self.x = self.reset()
        self.u_bound = 30 * np.array([-1, 1])

    def reset(self):
        self.t = 0
        if self.init_x:
            self.x = self.init_x
        else:
            self.x = np.zeros(self.x_dim)
        return self.x

    def render(self):
        pass

    def step(self, omega):
        if self.action_dim == 1:
            if type(omega) == np.ndarray:
                omega = omega[0]
        # 控制律
        x = self.x[0]
        x_dot = self.x[1]
        delta_x = x - self.xd
        delta_x_dot = x_dot - self.xd_dot
        a = 1
        b = 3
        u = - a * x - b - omega ** 2 * delta_x - 2 * omega * delta_x_dot + self.xd_dot2
        u_origin = u

        # 限幅
        if u > np.max(self.u_bound):
            u = np.max(self.u_bound)
        elif u < np.min(self.u_bound):
            u = np.min(self.u_bound)

        # 微分方程
        A = np.array([[0, 1], [a, 0]])
        B = np.array([0, 1])
        B_con = np.array([0,b])
        x_dot = np.dot(A, self.x) + np.dot(B, u) + B_con
        self.x += self.delta_t * x_dot
        self.t = self.t + self.delta_t

        # Reward Calculation
        reward = self.reward_design(u_origin, omega, (delta_x, delta_x_dot))

        info = {}
        info['action'] = u
        info['time'] = self.t
        info['u_ori'] = u_origin
        info['reward'] = reward
        if self.t > self.total_time:
            done = True
        else:
            done = False

        # Return
        return self.x, reward, done, info

    def reward_design(self, u, omega, delta):
        # 计算舵面奖励，可调参数界限penalty_bound
        u_norm = abs((u - np.mean(self.u_bound)) / abs(self.u_bound[0] - self.u_bound[1]) * 2)
        Penalty_bound = 0.75
        if u_norm < Penalty_bound:
            Satu_Penalty = 0
        else:
            if u_norm > 5:
                u_norm = 5
            Satu_Penalty = - 1000 * (np.exp(0.1 * (u_norm - Penalty_bound)) - 1)

        # w的奖励
        omega_Penalty = omega

        # 结束状态的奖励
        end_Penalty = 0
        if self.t > self.total_time:
            if abs(delta[0]) > 1:
                end_Penalty -= 1

        # 计算三部分reward，按照一定比例，可调比例
        reward =  Satu_Penalty/3.0 + 1 * omega_Penalty
        reward = reward / float(self.total_time / self.delta_t) + 10 * end_Penalty  # 归一化
        return reward
