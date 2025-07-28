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

int main() {
    mpfr_setup();
    int n = 5;
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> M(n, n);
    M.setRandom();
    std::cout << "M:\n" << M << std::endl;
    M = M.selfadjointView<Eigen::Upper>();
    std::cout << "M symmetric:\n" << M << std::endl;
}