#include "vectorize.hpp"

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
template<>
optMatrix<double> unvec<double, false>(const Eigen::Ref<const optVector<double>>& x, int n){
    return x.template reshaped(n, n);
}
template<>
optMatrix<std::complex<double>> unvec<double, true>(const Eigen::Ref<const optVector<double>>& x, int n){
    return unsplit(x).template reshaped(n, n);
}
