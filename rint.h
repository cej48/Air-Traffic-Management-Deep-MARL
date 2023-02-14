#include <limits>

// custom int that remain within range.
class RangeInt{
    int range_check(int val){
        if (val<this->min){
            return this->min;
        }
        if (val>this->max){
            return max;
        }
        return val;
    }
public:
    int value =0; 
    int min;
    int max;

    RangeInt(){
        this->value= 0;
        this->min = std::numeric_limits<int>::min(); // minimum value
        this->max = std::numeric_limits<int>::max();
    }
    RangeInt(int value, int min, int max){
        this->value = value;
        this->min = min;    
        this->max = max;
    }
    int operator+(int &value){
        return this->range_check(this->value+value);
    }
    int operator-(int &value){
        return this->range_check(this->value-value);
    }
    int operator*(int &value){
        return this->range_check(this->value*value);
    }
    int operator/(int &value){
        return this->range_check(this->value/value);
    }
    void operator+=(int &value){
        this->value = this->operator+(value);
    }
    void operator-=(int &value){
        this->value = this->operator-(value);
    }
    void operator*=(int &value){
        this->value = this->operator*(value);
    }
    void operator/=(int &value){
        this->value = this->operator/(value);
    }

};