#pragma once

class Weather{

    public:
        float windspeed;
        float direciton;
        float pressure;
        Weather(){
            this->windspeed = 0;
            this->direciton = 0;
            this->pressure = 0;
        };
        Weather(float windspeed, float direciton, float pressure){
            this->windspeed = windspeed;
            this->direciton = direciton;
            this->pressure = pressure;
        }
};