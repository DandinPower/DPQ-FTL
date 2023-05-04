from host.host_interface import HostInterface
from dpq.state import StatePreprocess
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()

TRACE_LENGTH = int(os.getenv('TRACE_LENGTH'))

EPOCH = 1
STEP = 100000

def main():
    statePreprocess = StatePreprocess()
    hostInterface = HostInterface()
    for i in range(EPOCH):
        state = hostInterface.NewEpisode()
        statePreprocess.NewEpisode()
        for j in range(STEP):
            inputs = statePreprocess.NewReq(state)
            print(inputs)
            reward, nextState = hostInterface.StepByAction(0)
            state = nextState

if __name__ == "__main__":
    main()