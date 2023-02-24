import sys
import numpy
sys.path.append("build/")

import PyATMSim

test = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0)
run = True

while run:
    run = test.step()
    for i in test.traffic:
        pass

