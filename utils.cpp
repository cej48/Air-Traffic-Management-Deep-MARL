#include "utils.h"

namespace Utils{

    double calculate_angle(arma::vec3 p1, arma::vec3 p2){
        return  p1[1]-p2[1] < 0 ? atan((p1[0]-p2[0])/
                    (p1[1]-p2[1])) 
                    
                    : PI + atan((p1[0]-p2[0])/
                    (p1[1]-p2[1]));
    }

    double rad_to_deg(double value){
        return (value*180)/PI;
    }
    double deg_to_rad(double value){
        return value *(PI/180);
    }

    float calculate_distance(arma::vec3 a, arma::vec3 b){
        return sqrt(pow(a[0] - b[0],2)
                +   pow(a[1] - b[1],2));
    }
}