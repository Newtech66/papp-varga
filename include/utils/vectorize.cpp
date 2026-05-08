#ifndef VECTORIZE_PAPP_VARGA_H
#define VECTORIZE_PAPP_VARGA_H
#include <complex>
#include <type_traits>
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename RealScalar>
optVector<RealScalar> split(const Eigen::Ref<const optVector<std::complex<RealScalar>>>& x){
    optVector<RealScalar> v(2 * x.size());
    v << x.real(), x.imag();
    return v;
}

template<typename RealScalar>
optVector<std::complex<RealScalar>> unsplit(const Eigen::Ref<const optVector<RealScalar>>& x){
    optVector<std::complex<RealScalar>> v(x.size() / 2);
    v.real() = x.head(x.size() / 2);
    v.imag() = x.tail(x.size() / 2);
    return v;
}

template<typename RealScalar>
optVector<RealScalar> vec(const Eigen::Ref<const optMatrix<std::complex<RealScalar>>>& X){
    return split<RealScalar>(X.template reshaped());
}

template<typename RealScalar>
optVector<RealScalar> vec(const Eigen::Ref<const optMatrix<RealScalar>>& X){
    return X.template reshaped();
}

template<typename RealScalar>
optMatrix<std::complex<RealScalar>> unvecComplex(const Eigen::Ref<const optVector<RealScalar>>& x, int n){
    return unsplit(x).template reshaped(n, n);
}

template<typename RealScalar>
optMatrix<RealScalar> unvecReal(const Eigen::Ref<const optVector<RealScalar>>& x, int n){
    return x.template reshaped(n, n);
}

#endif