#include "possemidefinite.hpp"
#include "vectorize.hpp"
#include "common_typedefs.hpp"
#include <Eigen/Cholesky>
#include <Eigen/LU>

template<typename prec_type>
RealPositiveSemidefinite<prec_type>::RealPositiveSemidefinite(const int n) : matrix_size(n){
    iden.setIdentity(matrix_size, matrix_size);
    P.setIdentity(matrix_size, matrix_size);
    this->barrier_parameter = matrix_size;
    this->num_params = matrix_size * matrix_size;
}

template<typename prec_type>
optVector<prec_type> RealPositiveSemidefinite<prec_type>::point() const{
    return vec<prec_type>(P);
}

template<typename prec_type>
void RealPositiveSemidefinite<prec_type>::updatePoint(const Eigen::Ref<const optVector<prec_type>>& p){
    P = unvecReal<prec_type>(p, matrix_size);
    jac_updated = false;
}

template<typename prec_type>
optVector<prec_type> RealPositiveSemidefinite<prec_type>::jacobian(){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return -vec<prec_type>(Pinv);
}

template<typename prec_type>
optVector<prec_type> RealPositiveSemidefinite<prec_type>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return vec<prec_type>(Pinv * unvecReal<prec_type>(v, matrix_size) * Pinv);
}

template<typename prec_type>
optVector<prec_type> RealPositiveSemidefinite<prec_type>::ihvp(const Eigen::Ref<const optVector<prec_type>>& v){
    return vec<prec_type>(P * unvecReal<prec_type>(v, matrix_size) * P);
}

template<typename prec_type>
ComplexPositiveSemidefinite<prec_type>::ComplexPositiveSemidefinite(const int n) : matrix_size(n){
    iden.setIdentity(matrix_size, matrix_size);
    P.setIdentity(matrix_size, matrix_size);
    this->barrier_parameter = matrix_size;
    this->num_params = 2 * matrix_size * matrix_size;
}

template<typename prec_type>
optVector<prec_type> ComplexPositiveSemidefinite<prec_type>::point() const{
    return vec<prec_type>(P);
}

template<typename prec_type>
void ComplexPositiveSemidefinite<prec_type>::updatePoint(const Eigen::Ref<const optVector<prec_type>>& p){
    P = unvecComplex<prec_type>(p, matrix_size);
    jac_updated = false;
}

template<typename prec_type>
optVector<prec_type> ComplexPositiveSemidefinite<prec_type>::jacobian(){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return -vec<prec_type>(Pinv);
}

template<typename prec_type>
optVector<prec_type> ComplexPositiveSemidefinite<prec_type>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return vec<prec_type>(Pinv * unvecComplex<prec_type>(v, matrix_size) * Pinv);
}

template<typename prec_type>
optVector<prec_type> ComplexPositiveSemidefinite<prec_type>::ihvp(const Eigen::Ref<const optVector<prec_type>>& v){
    return vec<prec_type>(P * unvecComplex<prec_type>(v, matrix_size) * P);
}

template<typename prec_type>
DiagonalPositiveSemidefinite<prec_type>::DiagonalPositiveSemidefinite(const int n) : matrix_size(n){
    p.setOnes(matrix_size);
    pinv.setOnes(matrix_size);
    this->barrier_parameter = matrix_size;
    this->num_params = matrix_size;
}

template<typename prec_type>
void DiagonalPositiveSemidefinite<prec_type>::updatePoint(const Eigen::Ref<const optVector<prec_type>>& p){
    this->p = p;
    pinv = p.inverse();
}

template class RealPositiveSemidefinite<double>;
template class ComplexPositiveSemidefinite<double>;
template class DiagonalPositiveSemidefinite<double>;