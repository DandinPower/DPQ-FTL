import matplotlib.pyplot as plt
import numpy as np

class TrainHistory:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.actions = []
        self.epsilon = []

    def AddHistory(self, datas):
        self.episodes.append(datas[0])
        self.rewards.append(datas[1])
        self.actions.append(datas[2])
        self.epsilon.append(datas[3])

    @staticmethod
    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods-1], res])
    
    def WriteHistory(self, path):
        with open(path, 'w') as file:
            # Iterate over the list
            for episode, reward in zip(self.episodes, self.rewards):
                # Write each item to a new line in the file
                file.write(f'Episode: {episode}, Reward: {reward}\n')

    def ShowHistory(self, path):
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.actions, color="C1", alpha=0.2)
        ax1.plot(self.rewards, color="C2", alpha=0.2)
        plt.title('Training Progress')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        mean_reward = self._moving_average(self.rewards, periods=5)
        lines = []
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])
        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilon, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)
        plt.savefig(path, dpi = 300)
        plt.clf()