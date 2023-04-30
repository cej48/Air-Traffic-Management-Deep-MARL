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


    float frame_length;
    RangeInt acceleration = RangeInt(1000, 1, 60); // rip

    int framerate;

    Atmosphere* environment;



    bool render;
    ATMInterface* interface;
    int traffic_ID=0;
    int traffic_timeout = 2000;

    void detect_closure_infringement();
    void detect_traffic_arrival();
    void verify_boundary_constraints();
    void spawn_aircraft();

private:
    float lattitude_min;
    float lattitude_max;
    float longitude_min;
    float longitude_max;
    unsigned int max_traffic_count = 25;
    int count =0;



public:

    int arrivals_sum=0;
    int total_steps=0;
    int total_infringements = 0;

    void copy_from_other(ATMSim *other);
    void calculate_rewards();

    std::vector<Traffic*> traffic = std::vector<Traffic*>();
    std::vector<Airport*> airports = std::vector<Airport*>();
    std::vector<float> observation= std::vector<float>(10);
    std::vector<float> action = std::vector<float>(10);
    ATMSim(ATMSim *other);
    ATMSim(ATMSim *other, bool render);
    ATMSim(std::string environment_meta, std::string airport_information, bool render, int framerate, float frame_length, int max_traffic_count);
    void detect_nearest_traffic(Traffic* traff, float angle, float distance_xy);

    int traffic_maximum;
    bool step();
    void reset();
    std::vector<float> get_rewards();
    void set_actions(std::vector<std::vector<float>> actions);
    bool skip_render = 0;
};