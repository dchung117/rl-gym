import random
from typing import Any
from collections import namedtuple, deque

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