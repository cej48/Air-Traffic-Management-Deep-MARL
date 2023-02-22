#include "utils.h"
#include "macro.h"

namespace Utils{

    double calculate_angle(Eigen::Vector3d p1, Eigen::Vector3d p2){
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

    float calculate_distance(Eigen::Vector3d a, Eigen::Vector3d b){
        return sqrt(pow(a[0] - b[0],2)
                +   pow(a[1] - b[1],2));
    }
}