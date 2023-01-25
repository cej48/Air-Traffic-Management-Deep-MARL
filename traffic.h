#pragma once
#include <iostream>
#include <vector>
#include <armadillo>
#include "weather.h"
#include <string>

class Traffic {
public:
    arma::Col<double> position;
    double speed;
    double heading;
    double rate_of_climb;
    double rate_of_turn;
    std::string destination; 
    std:: string callsign;
    bool instructed;
    

    Traffic(const double lattitude, const double longitude, 
            const double speed, const double rate_of_climb, 
            const double altitude, std::string destination, std::string callsign);

    void step(Weather* weather);
};