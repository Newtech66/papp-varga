#ifndef POINT_PAPP_VARGA_H
#define POINT_PAPP_VARGA_H
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
};

#endif