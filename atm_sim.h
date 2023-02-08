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

    float frame_length;
    int count =0;
    int acceleration;

    int framerate;

    Atmosphere* environment;
    std::vector<Traffic*> traffic = std::vector<Traffic*>();
    std::vector<Airport*> airports = std::vector<Airport*>();

    bool render;
    ATMInterface* interface;

    float calculate_distance(arma::vec3 a, arma::vec3 b);

    void detect_closure_infringement();
    void detect_traffic_arrival();


    public:

        ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length);
        int traffic_maximum;
        bool step();


    
};