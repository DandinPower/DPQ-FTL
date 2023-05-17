from .q_model import QModel, QModel_Deep
from .buffer import Transition
import numpy as np
import torch 
import random 
from dotenv import load_dotenv
import os
load_dotenv()

RNG = np.random.default_rng(100)
LR = float(os.getenv('LR'))
GAMMA = float(os.getenv('GAMMA'))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = os.getenv('MODEL_TYPE')

def GetModelByType(type):
    if MODEL_TYPE == 'deep':
        return QModel_Deep()
    elif MODEL_TYPE == 'normal':
        return QModel()

class ValueNetworks:
    def __init__(self):
        self.net = GetModelByType(MODEL_TYPE)
        self.net.to(DEVICE).train()
        self.targetNet = GetModelByType(MODEL_TYPE)
        self.targetNet.to(DEVICE).train()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.lossFn = torch.nn.SmoothL1Loss()

    def UpdateTargetNet(self):
        self.targetNet.load_state_dict(self.net.state_dict())

    def UpdateOptimizerLR(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def Optimize(self, batchData):
        self.optimizer.zero_grad()
        batchData = Transition(*zip(*batchData))
        state_batch = torch.cat(batchData.state)
        action_batch = torch.cat(batchData.action)
        reward_batch = torch.cat(batchData.reward)
        next_state_batch = torch.cat(batchData.nextState)
        model_output = self.net(state_batch)
        model_output = model_output.gather(1, action_batch.unsqueeze(1)).squeeze()
        target_output = self.targetNet(next_state_batch).max(1)[0]
        expected_q_values = (target_output * GAMMA) + reward_batch
        loss = self.lossFn(expected_q_values, model_output)
        loss.backward()
        self.optimizer.step()

    def GetModelAction(self, state, epsilon):
        if RNG.uniform() < epsilon:
            return random.choice([0, 1])
        else:
            with torch.no_grad():
                output = self.net(torch.tensor(state, dtype=torch.float32, device=DEVICE))
                _, predicted_labels = torch.max(output, dim=1)
                action = predicted_labels[0].cpu().numpy().item()
                return action
            
    def SaveWeight(self, path):
        torch.save(self.net.state_dict(), path)
        # print("Model saved successfully.")

    def LoadWeight(self, path):
        self.net.load_state_dict(torch.load(path))
        print("Model loaded successfully.")
