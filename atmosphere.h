#pragma once
#include "weather.h"
#include <iostream>
#include <vector>


class Atmosphere{
    
    public:
        std::vector<std::vector<std::vector<Weather>>> weather;

        Atmosphere(const int x_size, const int y_size, const int layers);
        
        void step();

};