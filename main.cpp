#include <iostream>
#include "atm_sim.h"
#include <string>

int main(){

    std::string environment_boundaries = "environment_boundaries.json";
    std::string airport_data = "airport_data.json";

    ATMSim simulator = ATMSim(environment_boundaries, airport_data, true);

    while(true) 
    {
        simulator.step();
    }

    return 0;
}