

import sys


import numpy
import RL
sys.path.append("build/")
sys.path.append('RL/')
from RL.ddpg_agent import DDPGAgent

import PyATMSim

test = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 0,0,0)

test_2 = PyATMSim.ATMSim(test, False)
run = True

agent = DDPGAgent(test, beta=0.005, gamma=0.99, sample_size=128)

while run:
    run = test.step()

    test_2.copy_from_other(test)
    
    print("Before: ")
    print(test.traffic[0].position)
    print(test_2.traffic[0].position)

    test_2.step()

    print("After: ")
    print(test.traffic[0].position)
    print(test_2.traffic[0].position)

