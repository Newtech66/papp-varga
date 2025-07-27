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
    MatrixXmp m(2, 2);
    VectorXmp v(2, 1);
    // m << 0.9542468245, 1, 1, -2.095893797;
    // v << -0.04359609171, 0.09589379665;
    m.setRandom();
    v.setRandom();
    // std::cout << "LLT solve:" << std::endl;
    // std::cout << m.llt().solve(v) << std::endl;
    std::cout << "m:" << std::endl;
    std::cout << m << std::endl; 
    std::cout << "m inverse:" << std::endl;
    MatrixXmp minv = m.colPivHouseholderQr().solve(MatrixXmp::Identity(2, 2));
    std::cout << minv << std::endl;
    std::cout << "m * minv:" << std::endl;
    std::cout << m * minv << std::endl;
    std::cout << "QR solve:" << std::endl;
    std::cout << m.colPivHouseholderQr().solve(v) << std::endl;
    std::cout << "minv * v:" << std::endl;
    std::cout << minv * v << std::endl;
}