from collections import deque
import random
from dotenv import load_dotenv
import os
load_dotenv()

MAX_QUEUE = int(os.getenv('MAX_QUEUE'))

class ReplayBuffer:
    # 初始化
    def __init__(self):
        self.memory = deque(maxlen= MAX_QUEUE)
    
    # 儲存一個step的資訊
    def Add(self, state, action, reward, nextState):
        self.memory.append((state, action, reward, nextState))

    # 取得batch資料
    def GetBatchData(self, _batchSize):
        data = random.sample(self.memory, _batchSize)
        return data
    
    # 取得目前長度
    def __len__(self):
        return len(self.memory)

    # 取得index的值
    def __getitem__(self, index):
        return self.memory[index]