#include "utils.h"
#include "macro.h"
#include <iostream>

namespace Utils{

    float calculate_angle(Eigen::Vector3f p1, Eigen::Vector3f p2){
        return  p1[1]-p2[1] < 0 ? 
                    atan((p1[0]-p2[0])/
                    (p1[1]-p2[1])) 
                    : PI + atan((p1[0]-p2[0])/
                    (p1[1]-p2[1]));
    }

    float rad_to_deg(float value){
        return (value*180)/PI;
    }
    float deg_to_rad(float value){
        return value*(PI/180);
    }

    float calculate_distance(Eigen::Vector3f a, Eigen::Vector3f b){
        return sqrt(pow(a[0] - b[0],2)
                +   pow(a[1] - b[1],2));
    }

}