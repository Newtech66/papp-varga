#ifndef CONE_DISPATCH_CONES_H
#define CONE_DISPATCH_CONES_H
#include "common_typedefs.hpp"
#include "cone.hpp"
#include "positive_semidefinite.hpp"
#include <memory>
#include <stdexcept>

template<typename prec_type, typename Derived>
bool cone_is_symmetric(std::string cone_id){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::isSymmetric();
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::isSymmetric();
    throw std::invalid_argument("Invalid cone name");
}

template<typename prec_type, typename Derived>
bool cone_is_complex(std::string cone_id){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::isComplex();
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::isComplex();
    throw std::invalid_argument("Invalid cone name");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_grad(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::grad(p, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::grad(p, cone_params);
    throw std::invalid_argument("Invalid cone name");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_hvp(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::hvp(p, q, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::hvp(p, cone_params);
    throw std::invalid_argument("Invalid cone name");
}

template<typename prec_type, typename Derived>
optVector<prec_type> cone_ihvp(std::string cone_id, const std::unique_ptr<ConeParameters>& cone_params, const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q){
    if(cone_id == "REALPSD")    return PositiveSemidefinite<prec_type, false>::ihvp(p, q, cone_params);
    if(cone_id == "COMPLEXPSD")    return PositiveSemidefinite<prec_type, true>::ihvp(p, cone_params);
    throw std::invalid_argument("Invalid cone name");
}

#endif