#pragma once
#include <eigen3/Eigen/Dense>
// #define PI 3.14159265

namespace Utils{

    float calculate_angle(Eigen::Vector3d p1, Eigen::Vector3d p2);

    float rad_to_deg(float value);
    float deg_to_rad(float value);

    float calculate_distance(Eigen::Vector3d a, Eigen::Vector3d b);
}