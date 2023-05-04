from .q_model import QModel
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
        self.optimizer.zero_grad()
        states = torch.tensor([d[0] for d in batchData], dtype=torch.float32).to(DEVICE)
        actions = torch.tensor([d[1] for d in batchData], dtype=torch.long).to(DEVICE)
        rewards = torch.tensor([d[2] for d in batchData], dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor([d[3] for d in batchData], dtype=torch.float32).to(DEVICE)
        model_output = self.net(states)
        target_output = self.targetNet(next_states).detach()
        model_output = model_output.gather(1, actions.unsqueeze(1)).squeeze()
        next_state_values, _ = target_output.max(dim=1)
        expected_q_values = (next_state_values * GAMMA) + rewards
        loss = self._loss(expected_q_values, model_output)
        loss.backward()
        self.optimizer.step()

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
