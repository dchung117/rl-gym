import numpy as np

class BlackJackAgent(object):
    """
    Epsilon greedy blackjack agent trained via Q-Learning

    :param n_actions - number of playable actions
    :dtype int
    :param lr - learning rate for updating Q-values
    :dtype float
    :param initial_eps - initial epsilon hyperparameter for exploration
    :dtype float
    :param eps_decay - rate at which exploration probability decreases during training
    :dtype float
    :param final_eps - final epsilon hyperparameter for exploration
    :dtype float
    :param gamma - discount factor to downweight Q-values of future states from current state
    :dtype float
    """
    def __init__(self, n_actions: int, lr: float, initial_eps: float, eps_decay: float, final_eps: float, gamma: float = 0.95) -> None:
        self.q_values = {}
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.eps = initial_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps

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

        if np.random.random() < self.eps:
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
        future_q = (not is_terminated)*self.q_values[next_obs].max()
        td = (reward + self.gamma*future_q - self.q_values[obs][action]) # temporal difference

        if obs not in self.q_values:
            self.q_values = np.zeros(self.n_actions)
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
        self.eps = max(self.final_eps, self.eps - self.eps_decay)
