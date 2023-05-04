from dotenv import load_dotenv
import numpy as np
import os
load_dotenv()

EPSILON = float(os.getenv('EPSILON'))
EPSILON_MIN = float(os.getenv('EPSILON_MIN'))
EPSILON_DECAY = float(os.getenv('EPSILON_DECAY'))
LR = float(os.getenv('LR'))
LR_MIN = float(os.getenv('LR_MIN'))
LR_DECAY = float(os.getenv('LR_DECAY'))

class HyperParameter:
    def __init__(self):
        self._epsilon = EPSILON
        self._lr = LR

    # 根據目前的episode調整epsilon
    def UpdateEpsilon(self, episode):
        delta = EPSILON - EPSILON_MIN
        self._epsilon = EPSILON_MIN + delta * np.exp(- episode / EPSILON_DECAY)

    # 根據目前的episode調整leaning_rate
    def UpdateLearningRate(self, episode):
        delta = LR - LR_MIN
        self._lr = LR_MIN + delta * np.exp(- episode / LR_DECAY)