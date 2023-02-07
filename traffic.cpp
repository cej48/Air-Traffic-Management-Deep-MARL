#include "traffic.h"
#include <math.h>

#define PI 3.14159265

Traffic::Traffic(double longitude, double lattitude, 
                const double speed, const double rate_of_climb, 
                const double altitude, std::string destination,
                std::string callsign, int frame_length)
{
    this->position = arma::vec({longitude, lattitude, altitude});
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
    this->frame_length = frame_length;
}

// 1 step 1/60th of a second
// at 60hz, 1 second = 60 steps.
void Traffic::step(Weather* weather)
{

    this->position[2] = this->target_altitude;

    this->position[0]+=sin(this->target_heading*(PI/180))*1e-5;
    this->position[1]+=cos(this->target_heading*(PI/180))*1e-5;

}
