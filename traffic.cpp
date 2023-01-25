#include "traffic.h"


Traffic::Traffic(const double lattitude, const double longitude, 
                const double speed, const double rate_of_climb, const double altitude)
{
    this->position = arma::Col<double>({lattitude, longitude, altitude});
}

void Traffic::step(Weather* weather)
{
    
}
