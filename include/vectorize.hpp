#ifndef VECTORIZE_PAPP_VARGA_H
#define VECTORIZE_PAPP_VARGA_H
#include <complex>
#include <type_traits>
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename prec_type>
optVector<prec_type> split(const Eigen::Ref<const optVector<std::complex<prec_type>>>& x);
template<typename prec_type>
optVector<std::complex<prec_type>> unsplit(const Eigen::Ref<const optVector<prec_type>>& x);
template<typename prec_type>
optVector<prec_type> vec(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& X);
template<typename prec_type>
optVector<prec_type> vec(const Eigen::Ref<const optMatrix<prec_type>>& X);
template<typename prec_type, bool is_complex>
std::conditional_t<is_complex, optMatrix<std::complex<prec_type>>, optMatrix<prec_type>>
unvec(const Eigen::Ref<const optVector<prec_type>>& x, int n);

#endif