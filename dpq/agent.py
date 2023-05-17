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
TRAIN_HISTORY_REPORT_PATH = os.getenv('TRAIN_HISTORY_REPORT_PATH')
MODEL_WEIGHT_PATH = os.getenv('MODEL_WEIGHT_PATH')
TRAIN_MODEL_WEIGHT = os.getenv('TRAIN_MODEL_WEIGHT')
SAVE_PERIOD = int(os.getenv('SAVE_PERIOD'))

TRACE_PATH = os.getenv('TRACE_PATH')
TRACE_LENGTH = int(os.getenv('TRACE_LENGTH'))
TRACE_2_PATH = os.getenv('TRACE_2_PATH')
TRACE_2_LENGTH = int(os.getenv('TRACE_2_LENGTH'))
LBA_FREQ_PATH = os.getenv('LBA_FREQ_PATH')
LBA_FREQ_2_PATH = os.getenv('LBA_FREQ_2_PATH')
BLOCK_NUM = int(os.getenv('BLOCK_NUM'))
BLOCK_NUM_2 = int(os.getenv('BLOCK_NUM_2'))

class MultiWorkloadScheduler:
    def __init__(self) -> None:
        self.hostInterfaces = []
        self.statePreProcesses = []
        self.numOfWorkload = 0
    
    def Reset(self):
        self.hostInterfaces.clear()
        self.statePreProcesses.clear()
        self.numOfWorkload = 0
    
    def AddWorkload(self, tracePath, traceLength, blockNum, lbaFreqPath):
        self.hostInterfaces.append(HostInterface(tracePath, traceLength, blockNum))
        self.statePreProcesses.append(StatePreprocess(tracePath, lbaFreqPath))
        self.numOfWorkload += 1

    def GetWorkload(self, episode):
        if self.numOfWorkload == 0:
            raise IndexError('There are no workload available')
        index = episode % self.numOfWorkload
        print(f'Workload Index: {index}')
        return self.hostInterfaces[index], self.statePreProcesses[index]

class Agent:
    def __init__(self) -> None:
        self.trainHistory = TrainHistory()
        self.replayBuffer = ReplayBuffer()
        self.valueNetworks = ValueNetworks()
        self.hyperParameter = HyperParameter()
        self.workloadScheduler = MultiWorkloadScheduler()
        self.workloadScheduler.AddWorkload(TRACE_PATH, TRACE_LENGTH, BLOCK_NUM, LBA_FREQ_PATH)
        self.workloadScheduler.AddWorkload(TRACE_2_PATH, TRACE_2_LENGTH, BLOCK_NUM_2, LBA_FREQ_2_PATH)
    
    def Episode(self, episode):
        hostInterface, statePreprocess = self.workloadScheduler.GetWorkload(episode)
        state = hostInterface.NewEpisode()
        statePreprocess.NewEpisode()
        state = statePreprocess.NewReq(state)
        rewardSum = 0
        for step in range(MAX_STEP):
            action = self.valueNetworks.GetModelAction(state, self.hyperParameter._epsilon)
            reward, nextState = hostInterface.StepByAction(action)
            rewardSum += reward
            nextState = statePreprocess.NewReq(nextState)
            self.replayBuffer.Add(state, action, reward, nextState)
            state = nextState
            if episode >= WARM_UP_EPISODES:
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
        self.trainHistory.WriteHistory(TRAIN_HISTORY_REPORT_PATH)

