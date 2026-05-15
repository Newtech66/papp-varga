#ifndef CONE_PARAMETERS_CONES_H
#define CONE_PARAMETERS_CONES_H
#include <string>
#include <concepts>

template<typename T>
concept ConeParameters = requires(T a){
    {a.isSymmetric()} -> std::same_as<bool>;
    {a.isComplex()} -> std::same_as<bool>;
    {a.coneName()} -> std::same_as<std::string>;
};

/*
 * This way you can have an std::vector<std::string> that can be passed to a dispatcher for the Cones,
 * avoiding virtual functions, but you can also have an std::vector<std::unique_ptr<ConeParameters>>
 * list for the parameters of each cone. So you pass the parameter list for each call to any function.
*/

#endif