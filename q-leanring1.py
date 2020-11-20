import time
import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")

env.reset()

done = False


LR = 0.1
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EV = 500

epsilon = 0.5
ST_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAY-ST_EPSILON_DECAY)

len_obs_space = 2 # currentSpeed direcction
# could by inputs + currentCorrelation?


DISC_OS_SIZE = [20] * len_obs_space

disc_os_win_size = (env.observation_space.high - env.observation_space.low) / DISC_OS_SIZE

n_action = 3
print(disc_os_win_size, env.observation_space.high)


q_table = np.random.uniform(low=-2, high=0, size=(DISC_OS_SIZE+[n_action]))


ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min':[], 'max':[]}



def get_discrete_state(state):
    discrete_state = (state-env.observation_space.low) / disc_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_rw = 0
    if episode % SHOW_EV == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, n_action)
            
        new_state, reward, done, _ = env.step(action)
        episode_rw += reward
        
        new_discrete_state = get_discrete_state(new_state)
        if render: 
            env.render()
            time.sleep(.01)


        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action,) ]

            new_q = (1-LR) * current_q + LR * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0]>=env.goal_position:
            q_table[discrete_state+(action,) ] = 0

        discrete_state= new_discrete_state
    if END_EPSILON_DECAY >= episode >= ST_EPSILON_DECAY:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_rw)

    if not episode % SHOW_EV:
        average_rw = sum(ep_rewards[-SHOW_EV:])/len(ep_rewards[-SHOW_EV:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_rw)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EV:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EV:]))

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()



