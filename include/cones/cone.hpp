#ifndef CONE_CONES_H
#define CONE_CONES_H
#include <concepts>
#include "common_typedefs.hpp"


template<typename T>
concept Cone = requires(T cone){
    {cone.isSymmetric()} -> std::same_as<bool>;
    {cone.isComplex()} -> std::same_as<bool>;
    {cone.coneId()} -> std::same_as<std::string>;
    {cone.coneName()} -> std::same_as<std::string>;
};

template<typename T, typename prec_type>
concept SymmetricCone = Cone<T> and
requires(T cone, const optVector<prec_type>& a, const optVector<prec_type>& b){
    {cone.hvp(a, b)} -> std::same_as<optVector<prec_type>>;
    {cone.ihvp(a, b)} -> std::same_as<optVector<prec_type>>;
    {cone.too(a, b)} -> std::same_as<optVector<prec_type>>;
    {cone.update_nt_scaling(a, b)};
    {cone.circle_product(a, b)};
    {cone.diamond_product(a, b)};
};

template<typename T, typename prec_type>
concept NonsymmetricCone = Cone<T> and
requires(T cone, const optVector<prec_type>& a, const optVector<prec_type>& b){
    {cone.hvp(a, b)} -> std::same_as<optVector<prec_type>>;
    {cone.ihvp(a, b)} -> std::same_as<optVector<prec_type>>;
    {cone.too(a, b)} -> std::same_as<optVector<prec_type>>;
};

/*
 * This way you can have an std::vector<std::string> that can be passed to a dispatcher for the Cones,
 * avoiding virtual functions, but you can also have an std::vector<std::unique_ptr<ConeParameters>>
 * list for the parameters of each cone. So you pass the parameter list for each call to any function.
*/

#endif