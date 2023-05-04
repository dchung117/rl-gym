import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    """
    Deep Q Network implementation (multi-layer perceptron).

    :param: n_obs - size of state vector
    :dtype: int
    :param: n_actions - number of actions to take
    :dtype: int
    """
    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(DeepQNetwork, self).__init__()
        self.layer_1 = nn.Linear(n_obs, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through deep Q network.

        :param: x - tensor representing state of environment (dim: (*, n_obs))
        :dtype: torch.Tensor
        :return - tensor of Q-values for action taken at given state input (dim: (*, n_actions))
        :rtype: torch.Tensor
        """
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
