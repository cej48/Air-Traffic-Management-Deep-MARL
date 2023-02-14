

class Heading{
public:
    double value;

    double operator+(double value){
        double out = int(this->value+value) % 360;

        if (out>360){
            out = out-360;
        }
        else if (out<0){
            out = out+360;
        }
        return out;
    }

    double operator-(double value){
        return this->operator+(-value);
    }
    void operator+=(double value){
        this->value = this->operator+(value);
    }

    void operator-=(double value){
        this->value = this->operator-(value);
    }
    void operator=(double value){
        this->value = value;
    }
    bool operator>(double value){
        return this->value>value;
    }
    bool operator<(double value){
        return this->value<value;
    }
};
