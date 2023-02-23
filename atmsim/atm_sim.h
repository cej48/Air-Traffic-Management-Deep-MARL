#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "traffic.h"
#include "atmosphere.h"
#include <eigen3/Eigen/Dense>
#include "airport.h"
#include "atm_interface.h"

class ATMSim{

    float lattitude_min;
    float lattitude_max;
    float longitude_min;
    float longitude_max;

    float frame_length;
    int count =0;
    unsigned int max_traffic_count = 50;

    RangeInt acceleration = RangeInt(60, 1, 60); // rip

    int framerate;

    Atmosphere* environment;
    std::vector<Traffic*> traffic = std::vector<Traffic*>();
    std::vector<Airport*> airports = std::vector<Airport*>();

    bool render;
    ATMInterface* interface;

    void detect_closure_infringement();
    double calculate_angle(Eigen::Vector3d p1, Eigen::Vector3d p2);
    void detect_traffic_arrival();

    void spawn_aircraft();

public:
    ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length);
    int traffic_maximum;
    bool step();


    
};