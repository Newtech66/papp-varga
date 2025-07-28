#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/MPRealSupport>

void mpfr_setup(){
    const int working_digits = 100;
    const int printing_digits = 20;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

using MatrixXmp = Eigen::Matrix<std::complex<mpfr::mpreal>, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXmp = Eigen::Vector<std::complex<mpfr::mpreal>, Eigen::Dynamic>;

int main() {
    mpfr_setup();
    int n;
    std::cin >> n;
    std::cout << std::lround<int>(std::sqrt(n)) << std::endl;
}