from host.host_interface import HostInterface
from .history import TrainHistory
from .buffer import ReplayBuffer
from .network import ValueNetworks
from .parameter import HyperParameter
from .state import StatePreprocess
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import os
load_dotenv()

EPISODES = int(os.getenv('EPISODES'))
MAX_STEP = int(os.getenv('MAX_STEP'))
WARM_UP_EPISODES = int(os.getenv('WARM_UP_EPISODES'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
UPDATE_RATE = int(os.getenv('UPDATE_RATE'))
TRAIN_HISTORY_PATH = os.getenv('TRAIN_HISTORY_PATH')
MODEL_WEIGHT_PATH = os.getenv('MODEL_WEIGHT_PATH')
TRAIN_MODEL_WEIGHT = os.getenv('TRAIN_MODEL_WEIGHT')
SAVE_PERIOD = int(os.getenv('SAVE_PERIOD'))

class Agent:
    def __init__(self) -> None:
        self.hostInterface = HostInterface()
        self.trainHistory = TrainHistory()
        self.replayBuffer = ReplayBuffer()
        self.valueNetworks = ValueNetworks()
        self.hyperParameter = HyperParameter()
        self.statePreprocess = StatePreprocess() 
    
    def Episode(self, episode):
        state = self.hostInterface.NewEpisode()
        self.statePreprocess.NewEpisode()
        state = self.statePreprocess.NewReq(state)
        rewardSum = 0
        for step in range(MAX_STEP):
            action = self.valueNetworks.GetModelAction(state, self.hyperParameter._epsilon)
            reward, nextState = self.hostInterface.StepByAction(action)
            rewardSum += reward
            nextState = self.statePreprocess.NewReq(nextState)
            self.replayBuffer.Add(state, action, reward, nextState)
            state = nextState
            if episode > WARM_UP_EPISODES:
                X = self.replayBuffer.GetBatchData(BATCH_SIZE)
                self.valueNetworks.Optimize(X)
            if step % UPDATE_RATE == 0:
                self.valueNetworks.UpdateTargetNet()
        self.trainHistory.AddHistory([episode, rewardSum, MAX_STEP, self.hyperParameter._epsilon])
        return rewardSum
    
    def Train(self):
        self.valueNetworks.LoadWeight(MODEL_WEIGHT_PATH)
        train_iter = tqdm(np.arange(EPISODES))
        for i in train_iter:
            rewardSum = self.Episode(i)
            self.hyperParameter.UpdateEpsilon(i)
            if i >= WARM_UP_EPISODES:
                self.hyperParameter.UpdateLearningRate(i - WARM_UP_EPISODES + 1)
                self.valueNetworks.UpdateOptimizerLR(self.hyperParameter._lr)
            train_iter.set_postfix_str(f"reward_sum: {rewardSum}")
            if i % SAVE_PERIOD == 0:
                self.valueNetworks.SaveWeight(f'{TRAIN_MODEL_WEIGHT}_{i}.pth')
        self.valueNetworks.SaveWeight(f'{TRAIN_MODEL_WEIGHT}_finish.pth')
        self.trainHistory.ShowHistory(TRAIN_HISTORY_PATH)

