#ifndef CONE_DISPATCH_CONES_H
#define CONE_DISPATCH_CONES_H
#include "common_typedefs.hpp"
#include "cone_parameters.hpp"
#include <memory>
#include <stdexcept>
#include <string>

template<typename prec_type, bool is_complex> class PositiveSemidefinite;

template<typename prec_type, typename Derived>
bool cone_is_symmetric(std::string cone_id){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::isSymmetric();
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::isSymmetric();
    throw std::invalid_argument("Attempt to call is_symmetric on " + cone_id + " failed!");
}

template<typename prec_type, typename Derived>
bool cone_is_complex(std::string cone_id){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::isComplex();
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::isComplex();
    throw std::invalid_argument("Attempt to call is_complex on " + cone_id + " failed!");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_grad(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::grad(p, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::grad(p, cone_params);
    throw std::invalid_argument("Attempt to call grad on " + cone_id + " failed!");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_hvp(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::hvp(p, q, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::hvp(p, cone_params);
    throw std::invalid_argument("Attempt to call hvp on " + cone_id + " failed!");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_ihvp(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::ihvp(p, q, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::ihvp(p, cone_params);
        throw std::invalid_argument("Attempt to call ihvp on " + cone_id + " failed!");
}

#endif