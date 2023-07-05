import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = "history/train/report/multi-workload_half_v1.txt"
SAVE_PATH = 'multi-half.png'
MOVING_AVERAGE_WINDOW = 20

def read_data(filename):
    episodes = []
    workloads = []
    rewards = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(',')
                episode = int(parts[0].split(':')[1].strip())
                workload = int(parts[1].split(':')[1].strip())
                reward = float(parts[3].split(':')[1].strip())
                episodes.append(episode)
                workloads.append(workload)
                rewards.append(reward)

    return episodes, workloads, rewards

def plot_rewards(episodes, workloads, rewards):
    plt.figure(figsize=(8, 6))
    workloads_unique = list(set(workloads))

    for wl in workloads_unique:
        wl_rewards = [reward for episode, reward in zip(episodes, rewards) if workloads[episode] == wl]
        aligned_episodes = list(range(len(wl_rewards)))

        # Calculate moving average
        moving_avg = np.convolve(wl_rewards, np.ones(MOVING_AVERAGE_WINDOW), 'valid') / MOVING_AVERAGE_WINDOW

        # Adjust the aligned episodes to match the moving average size
        aligned_episodes = aligned_episodes[MOVING_AVERAGE_WINDOW - 1:]

        plt.plot(aligned_episodes, moving_avg, label=f'Workload {wl}')

    plt.xlabel('Aligned Episode')
    plt.ylabel('Reward')
    plt.title('Workload Rewards (Moving Average, Window Size = {})'.format(MOVING_AVERAGE_WINDOW))
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_PATH, dpi=300)

episodes, workloads, rewards = read_data(FILE_PATH)
plot_rewards(episodes, workloads, rewards)
