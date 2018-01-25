from DQN import DeepQNetwork
from SmallStateControlDis import SSCPENV
import matplotlib.pyplot as plt
from DuelDQN import DuelingDQN as DQ

if __name__ == "__main__":
    # maze game
    state_track = []
    action_track = []
    time_track = []
    action_ori_track = []
    omega_track = []
    env = SSCPENV()
    RL = DQ(env.n_action, 2,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      replace_target_iter=500,
                      memory_size=3000,
                      e_greedy_increment=0.00003,
                      batch_size = 256,
                      output_graph=False
                      )
    step = 0

    episodes = 500
    for episode in range(episodes):
        ep_reward = 0
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)
            if episode == episodes - 1:
                state_track.append(observation.copy())
                action_track.append(info['action'])
                time_track.append(info['time'])
                action_ori_track.append(info['u_ori'])
                omega_track.append(action)
            ep_reward += reward
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print('Episode:', episode+1, ' ep_reward: %.4f' % ep_reward, 'epsilon: %.3f'%RL.epsilon)
    # end of game
    print('game over')
    print(ep_reward)
    plt.figure(1)
    plt.plot(time_track, [x[0] for x in state_track])
    plt.grid()
    plt.title('x')

    #
    plt.figure(2)
    plt.plot(time_track, action_track)
    plt.plot(time_track, action_ori_track)
    plt.title('action')
    plt.grid()

    plt.figure(3)
    plt.plot(time_track, omega_track)
    plt.title('omega')
    plt.grid()
    plt.show()