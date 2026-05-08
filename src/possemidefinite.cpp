#include "possemidefinite.hpp"
#include "vectorize.hpp"
#include "common_typedefs.hpp"
#include <Eigen/Cholesky>
#include <Eigen/LU>

template<typename prec_type, bool is_complex>
PositiveSemidefinite<prec_type, is_complex>::PositiveSemidefinite(const int n) : matrix_size(n){
    iden.setIdentity(matrix_size, matrix_size);
    P.setIdentity(matrix_size, matrix_size);
    this->barrier_parameter = matrix_size;
    if(is_complex)  this->num_params = 2 * matrix_size * matrix_size;
    else    this->num_params = matrix_size * matrix_size;
}

template<typename prec_type, bool is_complex>
optVector<prec_type> PositiveSemidefinite<prec_type, is_complex>::point() const{
    return vec<prec_type>(P);
}

template<typename prec_type, bool is_complex>
void PositiveSemidefinite<prec_type, is_complex>::updatePoint(const Eigen::Ref<const optVector<prec_type>>& p){
    P = unvec<prec_type, is_complex>(p, matrix_size);
    jac_updated = false;
}

template<typename prec_type, bool is_complex>
optVector<prec_type> PositiveSemidefinite<prec_type, is_complex>::jacobian(){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return -vec<prec_type>(Pinv);
}

template<typename prec_type, bool is_complex>
optVector<prec_type> PositiveSemidefinite<prec_type, is_complex>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
    if(!jac_updated){
        Pinv = P.ldlt().solve(iden);
        jac_updated = true;
    }
    return vec<prec_type>(Pinv * unvec<prec_type, is_complex>(v, matrix_size) * Pinv);
}

template<typename prec_type, bool is_complex>
optVector<prec_type> PositiveSemidefinite<prec_type, is_complex>::ihvp(const Eigen::Ref<const optVector<prec_type>>& v){
    return vec<prec_type>(P * unvec<prec_type, is_complex>(v, matrix_size) * P);
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

template class PositiveSemidefinite<double, false>;
template class PositiveSemidefinite<double, true>;
template class DiagonalPositiveSemidefinite<double>;