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

using MatrixXmp = Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXmp = Eigen::Vector<mpfr::mpreal, Eigen::Dynamic>;

int main() {
    mpfr_setup();
    mpfr::mpreal x("0.0001");
    mpfr::mpreal y("0.0002");
    std::swap(x, y);
    std::cout << x << std::endl;
    std::cout << y << std::endl;
}