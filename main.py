import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":
    # cartpole environment
    env = gym.make("CartPole-v1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")