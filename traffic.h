#pragma once
#include <iostream>
#include <vector>
#include <armadillo>
#include "weather.h"
#include <string>
#include "airport.h"
#include "heading.h"

class Traffic {

    void verify_constraints();

    void adjust_params();

public:
    arma::vec3 position;
    double speed;
    Heading heading;
    Airport* destination; 
    std:: string callsign;

    bool infringement=false;

    double rate_of_climb;
    double rate_of_turn;
    double rate_of_speed;

    double target_heading;
    double target_speed;
    double target_altitude;

    int frame_length;

    Traffic(const double longitude, const double lattitude, 
            const double speed, const double rate_of_climb, 
            const double altitude, Airport* destination, 
            std::string callsign, int framerate);

    void step(Weather* weather);
};