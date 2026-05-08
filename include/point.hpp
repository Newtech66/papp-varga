#ifndef POINT_PAPP_VARGA_H
#define POINT_PAPP_VARGA_H
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename prec_type>
struct Point{
    // should be private probably but whatever
    optVector<prec_type> x, y, z, s;
    prec_type kap, tau, theta;
    Point<prec_type>& operator+=(const Point<prec_type>& other);
    friend Point<prec_type> operator+(Point<prec_type> lhs, Point<prec_type>& other){
        lhs += other;
        return lhs;
    }
    Point<prec_type>& operator-=(const Point<prec_type>& other);
    friend Point<prec_type> operator-(Point<prec_type> lhs, Point<prec_type>& other){
        lhs -= other;
        return lhs;
    }
    Point<prec_type>& operator/=(const prec_type& other);
    friend Point<prec_type> operator/(Point<prec_type> lhs, prec_type& other){
        lhs /= other;
        return lhs;
    }
};

template<typename prec_type>
Point<prec_type>& Point<prec_type>::operator+=(const Point<prec_type>& other){
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    this->s += other.s;
    this->kap += other.kap;
    this->tau += other.tau;
    this->theta += other.theta;
    return *this;
}

template<typename prec_type>
Point<prec_type>& Point<prec_type>::operator-=(const Point<prec_type>& other){
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    this->s -= other.s;
    this->kap -= other.kap;
    this->tau -= other.tau;
    this->theta -= other.theta;
    return *this;
}

template<typename prec_type>
Point<prec_type>& Point<prec_type>::operator/=(const prec_type& other){
    this->x /= other;
    this->y /= other;
    this->z /= other;
    this->s /= other;
    this->kap /= other;
    this->tau /= other;
    this->theta /= other;
    return *this;
}

#endif