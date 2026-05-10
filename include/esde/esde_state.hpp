#ifndef POINT_PAPP_VARGA_H
#define POINT_PAPP_VARGA_H
#include "common_typedefs.hpp"
#include "problem_data.hpp"

template<typename prec_type>
struct ESDEState{
public:
    optVector<prec_type> x, y, z, s;
    prec_type kap, tau, theta;
    const ProblemData<prec_type>& problem_data;
    ESDE(const ProblemData<prec_type>& data){
        problem_data = std::make_unique(data);
        set_init_point();
    }
    void set_init_point();
};

#endif