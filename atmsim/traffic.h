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
    Eigen::Vector3f position;
    float speed;
    Heading heading;
    Airport* destination; 
    std:: string callsign;

    bool infringement=false;

    float rate_of_climb;
    float rate_of_turn;
    float rate_of_speed;

    float target_heading;
    float target_speed;
    float target_altitude;

    Heading destination_hdg;

    float reward;
    int frame_length;

    Traffic(Traffic *other);
    Traffic(const float longitude, const float lattitude, 
            const float speed, const float rate_of_climb, 
            const float altitude, Airport* destination, 
            std::string callsign, int framerate, int ID);

    std::vector<float>  get_observation();
    void set_actions(std::vector<float> actions);
    void step(Weather *weather);
};