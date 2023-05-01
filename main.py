from tqdm import tqdm
import gymnasium as gym

from agent import BlackJackAgent

if __name__ == "__main__":
    # hyperparameters
    lr = 0.01
    n_episodes = 100_000
    eps_init = 1.0
    eps_decay = eps_init / (n_episodes/2)
    eps_final = 0.1

    # blackjack env
    env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")
    n_actions = env.action_space.n

    agent = BlackJackAgent(n_actions=n_actions, lr=lr, eps_initial=eps_init, eps_decay=eps_decay, eps_final=eps_final)

    # wrapper to track episode statistics for Blackjack
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for i in tqdm(range(n_episodes)):
        # reset environment
        obs, info = env.reset()
        done = False

        while not done:
            # Take action (epsilon-greedy)
            action = agent.get_action(obs)
            next_obs, reward, is_terminated, is_truncated, info = env.step(action)

            # Update agent's Q-table
            agent.update(obs, action, reward, is_terminated, next_obs)

            # Update done and current observation
            done = is_terminated or is_truncated
            obs = next_obs

        # update learning rate
        agent.decay_eps()

    # Save Q-values and training error
    agent.save()