#pragma once
#include <armadillo>
#define PI 3.14159265

namespace Utils{

    double calculate_angle(arma::vec3 p1, arma::vec3 p2);

    double rad_to_deg(double value);
    double deg_to_rad(double value);

    float calculate_distance(arma::vec3 a, arma::vec3 b);
}