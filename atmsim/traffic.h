#pragma once
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "weather.h"
#include <string>
#include "airport.h"
// #include "heading.h"

class Traffic {

    void verify_constraints();

    void adjust_params();

public:
    int ID;
    int scale_speed=15;
    bool terminated = false;
    Eigen::Vector3f position;
    float speed;
    Heading heading;
    Airport* destination; 
    std::string callsign;

    bool infringement=false;

    float rate_of_climb;
    float rate_of_turn;
    float rate_of_speed;

    float target_heading;
    float target_speed;
    float target_altitude;

    Heading destination_hdg;

    float reward=0;
    int frame_length;

    Traffic(Traffic *other);
    Traffic(float longitude, float lattitude, 
            float speed, float rate_of_climb, 
            float altitude, Airport* destination, 
            std::string callsign, int framerate, int ID);

    std::vector<float>  get_observation();
    void set_actions(std::vector<float> actions);
    void step(Weather *weather);
};