#pragma once
#include <eigen3/Eigen/Dense>
// #define PI 3.14159265

namespace Utils{

    double calculate_angle(Eigen::Vector3d p1, Eigen::Vector3d p2);

    double rad_to_deg(double value);
    double deg_to_rad(double value);

    float calculate_distance(Eigen::Vector3d a, Eigen::Vector3d b);
}