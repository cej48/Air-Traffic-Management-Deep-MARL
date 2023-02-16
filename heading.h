#pragma once


class Heading{
public:
    double value;
    Heading(){
        this->value = 0;
    }
    Heading(double value){
        this->value= value;
    }

    double operator+(double value){
        double out = int(this->value+value) % 360;
        if (out<0){
            out = out+360;
        }
        return out;
    }

    double operator-(double value){
        return this->operator+(-value);
    }
    double operator-(Heading other){
        return this->operator-(other.value);
    }
    double operator+(Heading other){
        return this->operator+(other.value);
    }
    void operator+=(double value){
        this->value = this->operator+(value);
    }

    void operator-=(double value){
        this->value = this->operator-(value);
    }
    void operator=(double value){
        this->value = value;
        this->operator+=(0); // force range check.
    }
    bool operator>(double value){
        return this->value>value;
    }
    bool operator<(double value){
        return this->value<value;
    }

    bool in_range(double range, double fix){
        if (fix-range/2 < this->value && fix+range/2 > this->value){
            return true;
        }
        return false;
    }
};
