import random
from typing import Any, Optional
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

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

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor | None, reward: torch.Tensor) -> None:
        """
        Save new transition into memory buffer.

        :param: state - current environment state
        :dtype: torch.Tensor
        :param: action - index of action
        :dtype: torch.Tensor
        :param: next_state - potential next state after taking action at state
        :dtype: torch.Tensor | None
        :param: reward - reward after state/action pair
        :dtype: torch.Tensor
        :return: None
        :rtype: None
        """
        self.memory.append(Transition(state=state, action=action, next_state=next_state, reward=reward))

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

    if len(t) >= 100: # moving average over 100 episodes
        means = t.unfold(0, 100, 1).mean(dim=1).view(-1)
        means = torch.cat((torch.zeros(99), means), dim=0)
        plt.plot(means.numpy())

    if title:
        plt.savefig(f"tmp/{title}.png")
    else:
        plt.savefig("tmp/durations.png")