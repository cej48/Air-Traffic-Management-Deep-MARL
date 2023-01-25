#pragma once
#include <string>

class Airport{
    public:
        // Public to avoid excessive dataclass getter/setter
        double lattitude;
        double longitude;
        double runway_heading;
        std::string code;

        Airport(){
            double lattitude = 0;
            double longitude = 0;
            double runway_heading = 0;
            this->code = "";
        }
        Airport(double lattitude, double longitude, double runway_heading, std::string code){
            this->lattitude = lattitude;
            this->longitude = longitude;
            this->runway_heading = runway_heading;
            this->code = code;
        }

};