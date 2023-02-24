#pragma once


class Heading{
public:
    float value;
    Heading(){
        this->value = 0;
    }
    Heading(float value){
        this->value= value;
    }

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
