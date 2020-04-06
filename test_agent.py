import gym
import random
import torch
import numpy as np
from collections import deque

# import the agent
from dqn_agent import Agent

# create the lunar lander environment
env = gym.make('LunarLander-v2')

# define agent
agent = Agent(state_size=8, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# test the trained agent
state = env.reset()
env.render(mode='rgb_array')
for j in range(1000):
    action = agent.act(state)
    env.render(mode='rgb_array')
    state, reward, done, _ = env.step(action)
    if done:
        break 

env.close()