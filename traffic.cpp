#include "traffic.h"


Traffic::Traffic(const double lattitude, const double longitude, 
                const double speed, const double rate_of_climb, 
                const double altitude, std::string destination,
                std::string callsign)
{
    this->position = arma::Col<double>({lattitude, longitude, altitude});
    this->speed = speed;
    this->rate_of_climb = rate_of_climb;
    this->destination = destination;
    this->callsign = callsign;
}

void Traffic::step(Weather* weather)
{
    
}
