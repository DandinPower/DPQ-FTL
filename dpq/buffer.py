from collections import deque, namedtuple
import numpy as np
import random
import torch
from dotenv import load_dotenv
import os
load_dotenv()

MAX_QUEUE = int(os.getenv('MAX_QUEUE'))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextState'))

class ReplayBuffer:
    # 初始化
    def __init__(self):
        self.capacity = MAX_QUEUE
        self.memory = []
        self.position = 0
        self.rng = np.random.default_rng()
    
    # 儲存一個step的資訊
    def Add(self, state, action, reward, nextState):
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        action = torch.tensor([action], device=DEVICE, dtype=torch.long)
        reward = torch.tensor([reward], device=DEVICE)
        nextState = torch.tensor(nextState, device=DEVICE, dtype=torch.float32)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, nextState)
        self.position = (self.position + 1) % self.capacity

    # 取得batch資料
    def GetBatchData(self, batchSize):
        idx = self.rng.choice(np.arange(len(self.memory)), batchSize, replace=False)
        res = []
        for i in idx:
            res.append(self.memory[i])
        return res
    
    # 取得目前長度
    def __len__(self):
        return len(self.memory)