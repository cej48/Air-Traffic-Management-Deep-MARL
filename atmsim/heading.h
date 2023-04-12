#pragma once
#include <iostream>

class Heading{
public:
    float value;
    Heading(){
        this->value = 0;
    }
    Heading(float value){
        this->value= value;
    }
    // hdg 0, value = 200 
    float difference(float value){
        float out = (int)(value - this->value)%360;
        out = (out < -180) ? 360+out : out;
        out = (out > 180) ? -(360-out) : out;
        return out;
    }
    // 0 - 270
    float operator+(float value){
        float out = int(this->value+value) % 360;
        if (out<0){
            out = out+360;
        }
        return out;
    }

    float operator-(float value){
        
        return this->operator+(-value);
    }
    float operator-(Heading other){
        return this->operator-(other.value);
    }
    float operator+(Heading other){
        return this->operator+(other.value);
    }
    void operator+=(float value){
        this->value = this->operator+(value);
    }

    void operator-=(float value){
        this->value = this->operator-(value);
    }
    void operator=(float value){
        this->value = value;
        this->operator+=(0); // force range check.
    }
    void operator=(Heading other){
        this->value = other.value;
    }
    bool operator>(float value){
        return this->value>value;
    }
    bool operator<(float value){
        return this->value<value;
    }
    

    bool in_range(float range, float fix){
        if (fix-range/2 < this->value && fix+range/2 > this->value){
            return true;
        }
        return false;
    }
};
