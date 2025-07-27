#ifndef POINT_PAPP_VARGA_H
#define POINT_PAPP_VARGA_H
#include <Eigen/Dense>

template<typename RealScalar>
struct PointXYS{
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    // should be private probably but whatever
    Vector x, y, s;
};

template<typename RealScalar>
struct Point{
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    // should be private probably but whatever
    Vector x, y, z, s;
    RealScalar kap, tau, theta;
    Point<RealScalar>& operator+=(const Point<RealScalar>& other);
    friend Point<RealScalar> operator+(Point<RealScalar> lhs, Point<RealScalar>& other){
        lhs += other;
        return lhs;
    }
    Point<RealScalar>& operator-=(const Point<RealScalar>& other);
    friend Point<RealScalar> operator-(Point<RealScalar> lhs, Point<RealScalar>& other){
        lhs -= other;
        return lhs;
    }
    Point<RealScalar>& operator/=(const RealScalar& other);
    friend Point<RealScalar> operator/(Point<RealScalar> lhs, RealScalar& other){
        lhs /= other;
        return lhs;
    }
};

template<typename RealScalar>
Point<RealScalar>& Point<RealScalar>::operator+=(const Point<RealScalar>& other){
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    this->s += other.s;
    this->kap += other.kap;
    this->tau += other.tau;
    this->theta += other.theta;
    return *this;
}

template<typename RealScalar>
Point<RealScalar>& Point<RealScalar>::operator-=(const Point<RealScalar>& other){
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    this->s -= other.s;
    this->kap -= other.kap;
    this->tau -= other.tau;
    this->theta -= other.theta;
    return *this;
}

template<typename RealScalar>
Point<RealScalar>& Point<RealScalar>::operator/=(const RealScalar& other){
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