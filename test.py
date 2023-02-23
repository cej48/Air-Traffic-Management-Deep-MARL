import sys
import numpy
sys.path.append("build/")

import MyLib

test = MyLib.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0)

while True:
    test.step()
    print(test.traffic[0].position)