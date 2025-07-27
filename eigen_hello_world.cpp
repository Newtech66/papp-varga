#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/MPRealSupport>

void mpfr_setup(){
    const int working_digits = 100;
    const int printing_digits = 10;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

using MatrixXmp = Eigen::Matrix<std::complex<mpfr::mpreal>, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXmp = Eigen::Vector<std::complex<mpfr::mpreal>, Eigen::Dynamic>;

VectorXmp func(const Eigen::Ref<const Eigen::Vector<mpfr::mpreal, Eigen::Dynamic>>& x){
    VectorXmp v(2);
    v.real() = x.head(2);
    v.imag() = x.tail(2);
    return v;
}

int main() {
    mpfr_setup();
    Eigen::Vector<mpfr::mpreal, Eigen::Dynamic> x(4);
    x.setRandom();
    func(x);
}