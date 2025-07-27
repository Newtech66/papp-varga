#ifndef VECTORIZE_PAPP_VARGA_H
#define VECTORIZE_PAPP_VARGA_H
#include <complex>
#include <type_traits>
#include <Eigen/Core>

namespace Vectorize{
    template<typename RealScalar>
    Eigen::Vector<RealScalar, Eigen::Dynamic> split(const Eigen::Ref<const Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic>>& x){
        Eigen::Vector<RealScalar, Eigen::Dynamic> v(2 * x.size());
        v.head(x.size()) = x.real();
        v.tail(x.size()) = x.imag();
        return v;
    }

    template<typename RealScalar>
    Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> unsplit(const Eigen::Ref<const Eigen::Vector<RealScalar, Eigen::Dynamic>>& x){
        Eigen::Vector<std::complex<RealScalar>, Eigen::Dynamic> v(x.size() / 2);
        v.real() = x.head(x.size() / 2);
        v.imag() = x.tail(x.size() / 2);
        return v;
    }

    template<typename Scalar>
    Eigen::Vector<Scalar, Eigen::Dynamic> _internal_vec(const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>& X){
        return X.template reshaped<Eigen::RowMajor>();
    }

    template<typename RealScalar>
    Eigen::Vector<RealScalar, Eigen::Dynamic> vec(const Eigen::Ref<const Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic>>& X){
        return Vectorize::split<RealScalar>(Vectorize::_internal_vec(X));
    }

    template<typename RealScalar>
    Eigen::Vector<RealScalar, Eigen::Dynamic> vec(const Eigen::Ref<const Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>>& X){
        return Vectorize::_internal_vec(X);
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _internal_unvec(const Eigen::Ref<const Eigen::Vector<Scalar, Eigen::Dynamic>>& x){
        int n = std::lround<int>(x.size());
        return x.template reshaped<Eigen::RowMajor>(n, n);
    }

    template<typename RealScalar, bool IsComplex, std::enable_if_t<IsComplex, bool> = true>
    Eigen::Matrix<std::complex<RealScalar>, Eigen::Dynamic, Eigen::Dynamic> unvec(const Eigen::Ref<const Eigen::Vector<RealScalar, Eigen::Dynamic>>& x){
        return Vectorize::_internal_unvec<std::complex<RealScalar>>(Vectorize::unsplit(x));
    }

    template<typename RealScalar, bool IsComplex, std::enable_if_t<!IsComplex, bool> = true>
    Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic> unvec(const Eigen::Ref<const Eigen::Vector<RealScalar, Eigen::Dynamic>>& x){
        return Vectorize::_internal_unvec(x);
    }
}

#endif