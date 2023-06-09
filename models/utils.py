import random
from typing import Any, Optional
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import torch

Transition = namedtuple("Transition", ("state", "action" "next_state", "reward"))

class ReplayBuffer(object):
    """
    Agent memory used to sample transitions to train the deep RL models.

    :param: memory_limit - maximum number of transitions to store at one time.
    :dtype: int
    """
    def __init__(self, memory_limit: int) -> None:
        self.memory = deque([], maxlen=memory_limit)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args: Any) -> None:
        """
        Save new transition into memory buffer.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> list:
        """
        Sample from the memory buffer.

        :param: batch_size - size of sample to return
        :dtype: int
        :return: sample of transitions from memory buffer (length = batch_size)
        :rtype: list
        """
        return random.sample(self.memory, batch_size)

def plot_durations(durations: list[int], title: Optional[str] = None, show_result: bool = False) -> None:
    """
    Plot the durations of each episode over the training.

    :param durations - list of episode durations
    :dtype: list[int]
    :param title - optional title for figure = None
    :dtype: str
    :param: show_result - flag of whether to show figure = False
    :dtype: bool
    """
    plt.figure(1)
    t = torch.tensor(durations, dtype=torch.float32)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(t.numpy())

    if len(durations) >= 100: # moving average over 100 episodes
        means = t.unfold(0, 1000, 1).mean(dim=1).view(-1)
        means = torch.cat((torch.zeros(99), means), dim=0)
        plt.plot(means.numpy())

    if title:
        plt.savefig(f"{title}.png")
    else:
        plt.savefig("durations.png")