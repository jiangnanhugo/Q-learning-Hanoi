''' Reinforcement learning of the Towers of Hanoi game.
Reference: Watkins and Dayan, "Q-Learning", Machine Learning, 8, 279-292 (1992).'''
import sys
import numpy as np
from environment import generate_reward_matrix
from QLearning import learn_Q, get_policy
from utils import plot_results
import argparse
np.random.seed(10086)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Hanoi-v0')
    parser.add_argument('--save_path', default='./save_model/', help='file path to store the trained model')
    parser.add_argument('--log_path', default='log/epi_steps.log', help='episode length (used for plotting)')
    parser.add_argument('--render', action="store_true", default=True)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=30000)
    parser.add_argument('--learn_times', type=int, default=100)
    parser.add_argument('--play_times', type=int, default=100)
    parser.add_argument('--num_pillars', type=int, default=3)
    parser.add_argument('--num_disks', type=int, default=3)
    parser.add_argument('--K', type=int, default=1, help="max depth of MDD")
    return parser.parse_args()


def play_average(policy, play_times=100):
    moves = np.zeros(play_times)
    for n in range(play_times):
        moves[n] = 0
        start_state = 0
        end_state = len(policy) - 1
        state = start_state
        while state != end_state:
            state = np.random.choice(policy[state])
            moves[n] += 1
    return np.mean(moves), np.std(moves)


def Q_performance(R, episodes, play_times=100):
    means = np.zeros(len(episodes))
    stds = np.zeros(len(episodes))
    for n, N_episodes in enumerate(episodes):
        print("{}".format(n), end=" ")
        sys.stdout.flush()
        Q = learn_Q(R, N_episodes=N_episodes)
        policy = get_policy(Q, R)
        means[n], stds[n] = play_average(policy, play_times)
    return means, stds


if __name__ == '__main__':
    args = get_arguments()
    R = generate_reward_matrix(args.num_disks)
    episodes = [0, 1, 10, 30, 60, 100, 300, 600, 1000, 5000, 10000, 50000]
    ####
    # Q_performance_average begin
    means_times = np.zeros((args.learn_times, len(episodes)))
    stds_times = np.zeros((args.learn_times, len(episodes)))
    for n in range(args.learn_times):
        means_times[n, :], stds_times[n, :] = Q_performance(R, episodes, play_times=args.play_times)
    means_averaged = np.mean(means_times, axis=0)
    stds_averaged = np.mean(stds_times, axis=0)
    # Q_performance_average end
    plot_results(episodes, means_averaged, stds_averaged, args.num_disks, "Images/toh_3_3.pdf")
