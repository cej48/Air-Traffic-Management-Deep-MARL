#include "traffic.h"
#include <math.h>

#define PI 3.14159265

Traffic::Traffic(double longitude, double lattitude, 
                const double speed, const double rate_of_climb, 
                const double altitude, Airport* destination,
                std::string callsign, int frame_length)
{
    this->position = arma::vec({longitude, lattitude, altitude});
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
    this->frame_length = frame_length;
}

// let's make 1 step 1 second.
void Traffic::step(Weather* weather)
{

    this->position[2] = this->target_altitude;
    this->position[0]+=sin(this->target_heading*(PI/180))*1/pow(60,3)*this->speed;
    this->position[1]+=cos(this->target_heading*(PI/180))*1/pow(60,3)*this->speed;
    this->speed = this->target_speed;
    // if (this->speed == 150){
    //     this->infringement = true;
    // }
    // else{this->infringement=false;}
}
