#ifndef PSD_PARAMETERS_CONES_H
#define PSD_PARAMETERS_CONES_H
#include <string>
#include <format>

class PSDParameters{
protected:
    static const bool is_symmetric = true;
    const int matrix_size;
    const bool is_complex;
    
public:
    PSDParameters(int matrix_size, bool is_complex): matrix_size(matrix_size), is_complex(is_complex){}
    static bool isSymmetric(){return is_symmetric;}
    int barrierParameter() const{return matrix_size;}
    int numVariables() const{
        if(is_complex)  return matrix_size * (matrix_size + 1);
        else    return matrix_size * (matrix_size + 1) / 2;
    }
    std::string coneName() const{
        if(is_complex){
            return std::format("Cone of {0} x {0} complex Hermitian positive-semidefinite matrices", matrix_size);
        }else{
            return std::format("Cone of {0} x {0} real symmetric positive-semidefinite matrices", matrix_size);
        }
    }
    bool isComplex() const{return is_complex;}
    int matrixSize() const{return matrix_size;}
};

#endif