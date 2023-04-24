import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from tqdm import tqdm
import gymnasium as gym

if __name__ == "__main__":
    # blackjack env
    env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")

    # reset env
    done = False
    obs, info = env.reset() # obs: (user_hand, dealer_faceup, user_has_ace_bool (i.e. 11 w/o busting))
    print("User total: ", obs[0])
    print("Dealer face-up: ", obs[1])
    print("User has usable ace-high?: ", obs[2])
    print(info)
    print()

    # take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("User total: ", obs[0])
    print("Dealer face-up: ", obs[1])
    print("User has usable ace-high?: ", obs[2])
    print("Reward: ", reward)
    print("Game over?: ", terminated)
    print("Time limit reached?: ", truncated)
    print(info)
    print()

    # reset game if over
    if terminated:
        obs, info = env.reset()
        print("Game over. Resetting.")
        print("User total: ", obs[0])
        print("Dealer face-up: ", obs[1])
        print("User has usable ace-high?: ", obs[2])
        print(info)
        print()