import PyATMSim


class ATMSim_wrap(PyATMSim.ATMSim):

# test = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 0,0,0)
    def __init__(boundaries : str, airports : str, render : bool, arg1 : int, arg2: int):
        super().init(boundaries, airports, render, arg1, arg2)