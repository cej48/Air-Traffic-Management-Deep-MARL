#pragma once
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "weather.h"
#include <string>
#include "airport.h"
#include "macro.h"
#define N_closest 4

class Traffic {

    void verify_constraints();

    void adjust_params();

    void get_and_sort_closest_distances();



public:
    int ID;
    int scale_speed=15;
    int start_count;
    bool terminated = false;
    bool silent_terminated = false;
    Eigen::Vector3f position;
    float speed;
    Heading heading;
    Airport* destination; 


    Traffic* null_traffic;
    std::vector<std::pair<float, Traffic*>> closest;
    // std::vector<float> closest_distances;
    // // Traffic* closest;
    // float closest_distance=0;

    // Traffic* second_closest;
    // float second_closest_distance=0;

    std::string callsign;

    bool infringement=false;
    bool prev_infringement = false;
    bool conflict_flag = false;


    bool potential_infringement = false;
    bool prev_potential_infringement = false;
    
    float rate_of_climb;
    float rate_of_turn;
    float rate_of_speed;

    float target_heading;
    float target_speed;
    float target_altitude;

    Heading destination_hdg;
    float distance_to;

    int lifespan;

    float reward=0;
    int frame_length;

    Traffic(Traffic *other);
    Traffic(float longitude, float lattitude, 
            float speed, float rate_of_climb, 
            float altitude, Airport* destination, 
            std::string callsign, int framerate, int ID
            , int start_count);

    std::vector<double>  get_observation();
    void set_actions(std::vector<float> actions);
    void clear_nearest();
    void step(Weather *weather);
};