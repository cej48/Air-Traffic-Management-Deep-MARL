#pragma once
#include <string>
#include <eigen3/Eigen/Dense>
#include "heading.h"

class Airport{
    public:
        // Public to avoid excessive dataclass getter/setter
        Eigen::Vector3d position;
        Heading runway_heading;
        std::string code;

        Airport(){

            Eigen::Vector3d position;
            Heading runway_heading(0);
            this->code = "";
        }
        Airport(float lattitude, float longitude, float runway_heading, std::string code){
            this->position[1] = lattitude;
            this->position[0] = longitude;
            this->position[2] = 0;
            this->runway_heading = runway_heading;
            this->code = code;
        }

};