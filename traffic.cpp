#include "traffic.h"


Traffic::Traffic(double longitude, double lattitude, 
                const double speed, const double rate_of_climb, 
                const double altitude, std::string destination,
                std::string callsign)
{
    this->position = arma::vec({longitude, lattitude, altitude});
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
}

// 1 step 1/60th of a second
// at 60hz, 1 second = 60 steps.
void Traffic::step(Weather* weather)
{
    this->position+=1e-5;
}
