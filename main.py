import math
import random
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
import torch

from agent import CartPoleAgent
from models.utils import plot_durations

BATCH_SIZE = 128
GAMMA = 0.99
EPS_INIT = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
N_EPISODES = 600

if __name__ == "__main__":
    # cartpole environment
    env = gym.make("CartPole-v1")

    # Initialize agent
    state, _ = env.reset()
    agent = CartPoleAgent(batch_size=BATCH_SIZE, gamma=GAMMA, eps_init=EPS_INIT, eps_end=EPS_END, eps_decay=EPS_DECAY,
        tau=TAU, lr=LR, n_obs=len(state), n_actions=env.action_space.n)

    episode_durations = []
    for i in range(N_EPISODES):
        # reset env, get current state, info from env
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in count():
            # Agent takes action (eps-greedy policy)
            action = agent.select_action(state)
            obs, reward, is_term, is_truncated, _ = env.step(action.item())

            # record transition
            reward = torch.tensor([reward])
            done = is_term or is_truncated
            if done:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            agent.replay_buffer.push(state, action, next_state, reward)

            # update current state
            state = next_state

            # train model
            agent.train()

            # soft-update target qnet
            agent.update_target_net(tau=TAU)

            # End episode if game over
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations, title=f"episode_{i}")
                break

    print("Training done.")
    plot_durations(episode_durations, title="end", show_result=True)


