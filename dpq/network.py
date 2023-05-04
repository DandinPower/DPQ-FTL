from .q_model import QModel
import numpy as np
import torch 
import random 
from dotenv import load_dotenv
import os
load_dotenv()

RNG = np.random.default_rng(100)
LR = float(os.getenv('LR'))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class ValueNetworks:
    def __init__(self):
        self.net = QModel().to(DEVICE).train()
        self.targetNet = QModel().to(DEVICE).train()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.lossFn = torch.nn.HuberLoss()

    def UpdateTargetNet(self):
        self.targetNet.load_state_dict(self.net.state_dict())

    def UpdateOptimizerLR(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def Optimize(self, batchData):
        pass

    def GetModelAction(self, state, epsilon):
        if RNG.uniform() < epsilon:
            return random.choice([0, 1])
        else:
            with torch.no_grad():
                output = self.net(torch.tensor(state, dtype=torch.float32, device=DEVICE))
                _, predicted_labels = torch.max(output, dim=1)
                return predicted_labels[0].cpu().numpy()
            
    def SaveWeight(self, path):
        torch.save(self.net.state_dict(), path)
        print("Model saved successfully.")

    def LoadWeight(self, path):
        self.net.load_state_dict(torch.load(path))
        print("Model loaded successfully.")
