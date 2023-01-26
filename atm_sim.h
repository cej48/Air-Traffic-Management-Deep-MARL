#pragma once
#include <iostream>
#include <vector>
#include "traffic.h"
#include "atmosphere.h"
#include <armadillo>
#include "airport.h"
#include "atm_interface.h"

class ATMSim{

    float lattitude_min;
    float lattitude_max;
    float longitude_min;
    float longitude_max;
    int framerate;

    Atmosphere* environment;
    std::vector<Traffic*> traffic = std::vector<Traffic*>();
    std::vector<Airport*> airports = std::vector<Airport*>();

    bool render;
    ATMInterface* interface;    

    public:

        ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate);
        int traffic_maximum;
        bool step();


    
};