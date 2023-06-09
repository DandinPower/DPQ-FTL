from sklearn.preprocessing import StandardScaler
from ftl.pretrain.lba_dict import GetLbaFreqDict
import pandas as pd
import numpy as np

class StatePreprocess:
    def __init__(self, tracePath, lbaFreqPath) -> None:
        self.tracePath = tracePath
        self.lbaFreqPath = lbaFreqPath
        self.Initialize()
    
    def Initialize(self):
        self.scalerLbaDiff = StandardScaler()
        self.scalerBytes = StandardScaler()
        self.prevLba = 0
        self.lbaFreqDict = GetLbaFreqDict(self.lbaFreqPath)
        self.Standardize()

    def Standardize(self):
        data = pd.read_csv(self.tracePath, header=None, dtype=np.float64).values
        data[1:, 2] = np.diff(data[:, 2])
        data[0, 2] = 0
        data[:, 2] = self.scalerLbaDiff.fit_transform(data[:, 2].reshape(-1, 1)).flatten()
        data[:, 3] = self.scalerBytes.fit_transform(data[:, 3].reshape(-1, 1)).flatten()
    
    def NewEpisode(self):
        self.prevLba = 0

    def NewReq(self, request):
        lbaDiff = request.lba - self.prevLba
        self.prevLba = request.lba
        bytes = request.bytes
        standardized_lba_diff = self.scalerLbaDiff.transform(np.array(lbaDiff).reshape(1, -1))
        standardized_bytes = self.scalerBytes.transform(np.array(bytes).reshape(1, -1))
        standardized_lba = np.array(self.lbaFreqDict[str(request.lba)]).reshape(1, -1)
        input_data = np.concatenate((standardized_lba, standardized_lba_diff, standardized_bytes), axis=1)
        return input_data