#include "vectorize.hpp"

template<>
optMatrix<double> unvec<double, false>(const Eigen::Ref<const optVector<double>>& x, int n){
    return x.template reshaped(n, n);
}
template<>
optMatrix<std::complex<double>> unvec<double, true>(const Eigen::Ref<const optVector<double>>& x, int n){
    return unsplit(x).template reshaped(n, n);
}
