#ifndef ESDE_STATE_ESDE_H
#define ESDE_STATE_ESDE_H
#include "common_typedefs.hpp"

/// @brief Stores the ESDE state (x, y, z, tau, s, kap).
/// @tparam prec_type Underlying precision type of the solver.
template<typename prec_type>
struct ESDEState{
public:
    optVector<prec_type> x;
    optVector<prec_type> y;
    optVector<prec_type> z;
    prec_type tau;
    optVector<prec_type> s;
    prec_type kap;
    ESDEState() = default;
    ESDEState<prec_type>& operator+=(const ESDEState<prec_type>& other){
        this->x += other.x;
        this->y += other.y;
        this->z += other.z;
        this->tau += other.tau;
        this->s += other.s;
        this->kap += other.kap;
        return *this;
    }
    // https://stackoverflow.com/questions/29492869/multiplication-operator-overloading
    ESDEState<prec_type> operator*(const prec_type& scalar) const{
        return ESDEState<prec_type>{scalar * x, scalar * y, scalar * z,
        scalar * tau, scalar * s, scalar * kap};
    }
    friend ESDEState<prec_type> operator*(prec_type scalar, const ESDEState<prec_type>& v) {
        return v * scalar;
    }
};

#endif