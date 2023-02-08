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

    this->position[2]+=this->rate_of_climb;
    this->position[0]+=sin(this->heading*(PI/180))*1/pow(60,3)*this->speed;
    this->position[1]+=cos(this->heading*(PI/180))*1/pow(60,3)*this->speed;
    this->speed = this->target_speed;

    this->heading +=this->rate_of_turn;

    if (this->heading > 360){ // move to func
        this->heading = 0;
    }
    else if (this->heading < 0){
        this->heading = 360;
    }

    double det = (this->heading)-(this->target_heading);
    if (det>180|| det<-180){
        det = -det;
    }
    if (abs(det) < 3){
        this->rate_of_turn = det;
    }else{
        this->rate_of_turn = ((det < 0 ) - (det > 0 )) *3;
    }

    det = this->position[2] - this->target_altitude;
    if (abs(det) < 20){
        this->rate_of_climb = det;
    }else{
        this->rate_of_climb = ((det < 0 ) - (det > 0 )) *20;
    }
}
