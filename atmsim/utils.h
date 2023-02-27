#pragma once
#include <eigen3/Eigen/Dense>
#include "traffic.h"

namespace Utils{

    float calculate_angle(Eigen::Vector3f p1, Eigen::Vector3f p2);

    float rad_to_deg(float value);
    float deg_to_rad(float value);

    float calculate_distance(Eigen::Vector3f a, Eigen::Vector3f b);

    template <class T>

    // Needs copy constructor.
    void deepcopy_pointy_vectors(std::vector<T*>* a, std::vector<T*>* b)
    {
        b->clear();
        for (unsigned int i=0; i<a->size(); i++)
        {
            b->push_back(new T(a->at(i)));
        }
    };

};