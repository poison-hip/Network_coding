import gymnasium as gym
import os
import time 
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sac_agent import SACAgent

scenario = 'Pendulum-v1'
env = gym.make(id= scenario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MEMORY_SIZE = 1000000

agent = SACAgent(beta=3e-4, gamma=0.99, tau=0.005, alpha=3e-4, state_dim=STATE_DIM, 
                 action_dim=ACTION_DIM, layer1_dim=64, layer2_dim=64, 
                 memo_capacity = MEMORY_SIZE, batch_size=256)

NUM_EPISODE = 100
NUM_STEP = 200
PLOT_REWARD = True


current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
if not os.path.exists(model):
    os.makedirs(model)
timestamp = time.strftime('%Y%m%d%H%M%S')

REWARD_BUFF = []
best_reward = -np.inf
for episode_i in range (NUM_EPISODE):
    state , other = env.reset()
    episode_reward = 0
    for step_i in range(NUM_STEP):
        action = agent.get_action(state)
        next_state, reward, done, trunc, info = env.step(action)
        agent.add_memory(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        agent.update()
        if done:
            break
    REWARD_BUFF.append(episode_reward)
    avg_reward = np.mean(REWARD_BUFF)

    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(agent.ActionNetwork.state_dict(),model + f'sac_actor_{timestamp}.pth')
        print(f"...saving model with best reward:{best_reward}")

    print(f"Episode {episode_i}", 'reward %.1f' % episode_reward, 'avg_reward %.1f' %avg_reward)

env.close()

if PLOT_REWARD:
    plt.plot(np.arange(len(REWARD_BUFF)), REWARD_BUFF,color='purple', alpha=0.5, label='Reward')
    plt.plot(np.arange(len(REWARD_BUFF)), gaussian_filter(REWARD_BUFF, sigma=5), color='purple', linewidth=2)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig(f"Reward-{scenario}-{timestamp}.png", format='png')
    plt.show()
