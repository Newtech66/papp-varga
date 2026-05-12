#ifndef POSITIVE_SEMIDEFINITE_PARAMETERS_CONES_H
#define POSITIVE_SEMIDEFINITE_PARAMETERS_CONES_H
#include "cone_parameters.hpp"
#include <string>

class PositiveSemidefiniteParameters : public ConeParameters{
protected:
    int matrix_size;
public:
    void parse_args(const std::string& args) override;
    int matrixSize(){return matrix_size;}
};

#endif