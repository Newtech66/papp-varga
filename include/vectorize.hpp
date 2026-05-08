#ifndef VECTORIZE_PAPP_VARGA_H
#define VECTORIZE_PAPP_VARGA_H
#include <complex>
#include <type_traits>
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename prec_type>
optVector<prec_type> split(const Eigen::Ref<const optVector<std::complex<prec_type>>>& x){
    optVector<prec_type> v(2 * x.size());
    v << x.real(), x.imag();
    return v;
}
template<typename prec_type>
optVector<std::complex<prec_type>> unsplit(const Eigen::Ref<const optVector<prec_type>>& x){
    optVector<std::complex<prec_type>> v(x.size() / 2);
    v.real() = x.head(x.size() / 2);
    v.imag() = x.tail(x.size() / 2);
    return v;
}
template<typename prec_type>
optVector<prec_type> vec(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& X){
    return split<prec_type>(X.template reshaped());
}
template<typename prec_type>
optVector<prec_type> vec(const Eigen::Ref<const optMatrix<prec_type>>& X){
    return X.template reshaped();
}
// template<typename prec_type>
// optMatrix<std::complex<prec_type>> unvecComplex(const Eigen::Ref<const optVector<prec_type>>& x, int n){
//     return unsplit<prec_type>(x).template reshaped(n, n);
// }
// template<typename prec_type>
// optMatrix<prec_type> unvecReal(const Eigen::Ref<const optVector<prec_type>>& x, int n){
//     return x.template reshaped(n, n);
// }
template<typename prec_type, bool is_complex>
std::conditional_t<is_complex, optMatrix<std::complex<prec_type>>, optMatrix<prec_type>> unvec(const Eigen::Ref<const optVector<prec_type>>& x, int n);

#endif