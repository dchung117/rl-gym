import pickle
import numpy as np
import torch
import torch.optim as optim

from models.qnet import DeepQNetwork
from models.utils import ReplayBuffer

class BlackJackAgent(object):
    """
    Epsilon greedy blackjack agent trained via Q-Learning

    :param n_actions - number of playable actions
    :dtype int
    :param lr - learning rate for updating Q-values
    :dtype float
    :param eps_initial - initial epsilon hyperparameter for exploration
    :dtype float
    :param eps_decay - rate at which exploration probability decreases during training
    :dtype float
    :param eps_final - final epsilon hyperparameter for exploration
    :dtype float
    :param gamma - discount factor to downweight Q-values of future states from current state
    :dtype float
    """
    def __init__(self, n_actions: int, lr: float, eps_initial: float, eps_decay: float, eps_final: float, gamma: float = 0.95) -> None:
        self.q_values = {}
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.eps = eps_initial
        self.eps_decay = eps_decay
        self.eps_final = eps_final

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Sample action from policy given observation.

        Takes random action w/ probability epsilon; otherwise, take the best action

        :param obs - observation tuple containing (sum_user_hand, dealer_card, usable_ace_high_available)
        :dtype tuple[int, int, bool]
        :return - index of action to take (i.e. action with highest Q-value)
        :rtype int
        """

        if (np.random.random() < self.eps) or (obs not in self.q_values):
            return np.random.choice(self.n_actions)

        return np.argmax(self.q_values[obs])

    def update(self, obs: tuple[int, int, bool], action: int, reward: float, is_terminated: bool, next_obs: tuple[int, int, bool]) -> None:
        """
        Update Q-value w/ state-action-reward-is_terminated_next_state array.

        :param obs - observation tuple containing (sum_user_hand, dealer_card, usable_ace_high_available)
        :dtype tuple[int, int, bool]
        :param action - index of action taken
        :dtype int
        :param reward - reward received taking action at current obs
        :dtype float
        :param is_terminated - flag indicating if game is over
        :dtype bool
        :param next_obs - next observation tuple (sum_user_hand, dealer_card, usable_ace_high_available)
        :dtype tuple[int, int, bool]
        :return None
        :rtype None
        """
        # Q-learning update
        future_q = 0
        if next_obs in self.q_values:
            future_q = (not is_terminated)*self.q_values[next_obs].max()
        td = reward + self.gamma*future_q
        if obs in self.q_values:
            td -= self.q_values[obs][action] # temporal difference

        if obs not in self.q_values:
            self.q_values[obs] = np.zeros(self.n_actions)
        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * td)

        # track td errors
        self.training_error.append(td)

    def decay_eps(self) -> None:
        """
        Decays epsilon hyperparameter throughout training.

        Decay follows a linear schedule until terminal epsilon value is reached

        :return None
        :rtype None
        """
        self.eps = max(self.eps_final, self.eps - self.eps_decay)

    def save(self) -> None:
        """
        Save the Q-table and the training errors.

        :return None
        :rtype None
        """
        with open("q_values.pkl", "wb") as f:
            pickle.dump(self.q_values, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open("training_error.pkl", "wb") as f:
            pickle.dump(self.training_error, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """
        Load the Q-table and the training errors.

        :return None
        :rtype None
        """
        with open("q_values.pkl", "rb") as f:
            self.q_values = pickle.load(f)
        with open("training_error.pkl", "rb") as f:
            self.training_error = pickle.load(f)

class CartPoleAgent(object):
    """
    Agent to train and play cartpole game via Deep Q Learning.

    :param: batch_size - size of sample to draw from replay buffer for training.
    :dtype: int
    :param: gamma - discount factor
    :dtype: float
    :param: eps_init - initial exploration hyperparameter for epsilon-greedy policy training
    :dtype: float
    :param: eps_end - final exploration hyperparameter for epsilon-greedy policy training
    :dtype: float
    :param: eps_decay - decay factor for epsilon hyperparameter (i.e. larger -> slower decay rate)
    :dtype: int
    :param: tau - update rate for Q-target network
    :dtype: float
    :param: lr - learning rate of AdamW optimizer
    :dtype: float
    :param: n_obs - size of state vector
    :dtype: int
    :param: n_actions - number of actions for agent
    :dtype: int
    """
    def __init__(self, batch_size: int, gamma: float, eps_init: float, eps_end: float, eps_decay: int,
        tau: float, lr: float, n_obs: int, n_actions: int, cuda: bool = False) -> None:
        self.batch_size = batch_size
        self.gamma = gamma

        self.eps = eps_init
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.tau = tau

        self.device = torch.device("cpu")
        if cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_net = DeepQNetwork(n_obs, n_actions).to(self.device)
        self.q_net_target = DeepQNetwork(n_obs, n_actions).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.optim = optim.AdamW(self.q_net.parameters(), lr=lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(memory_limit=10000)

        self.n_steps = 0