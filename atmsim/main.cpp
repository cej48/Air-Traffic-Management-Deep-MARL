#include <iostream>
#include "atm_sim.h"
#include <string>

int main(){
    std::string environment_boundaries = "environment_boundaries.json";
    std::string airport_data = "airport_data.json";

    ATMSim simulator = ATMSim(environment_boundaries, airport_data, true, 60, 5);

    bool running = true;
    int count=0;
    while(running) 
    {
        count++;
        running = simulator.step();
    }

    return 0;
}