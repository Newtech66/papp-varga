#include "cppad_mpreal.cpp"
#include <cppad/example/cppad_eigen.hpp>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MPRealSupport>

namespace Eigen {
 
template<> struct NumTraits<CppAD::AD<mpfr::mpreal>>
 : NumTraits<mpfr::mpreal> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef CppAD::AD<mpfr::mpreal> Real;
};
}

void mpfr_setup(){
    const int working_digits = 100;
    const int printing_digits = 10;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randomPSD(const Eigen::Ref<const Eigen::Vector<T, Eigen::Dynamic>>& e){
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M(e.size(), e.size());
    M.setRandom();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Q = M.colPivHouseholderQr().matrixQ();
    return Q.transpose() * e.asDiagonal() * Q;
}

int main(){
    mpfr_setup();
    int n = 100;
    int num_inputs = n * n;
    int num_outputs = n * n;
    Eigen::Vector<CppAD::AD<mpfr::mpreal>, Eigen::Dynamic> init_eigs(n);
    init_eigs.setRandom();
    init_eigs = (2 * init_eigs.array() + 1).matrix();
    Eigen::Matrix<CppAD::AD<mpfr::mpreal>, Eigen::Dynamic, Eigen::Dynamic> init_vector(n, n);
    init_vector = randomPSD<CppAD::AD<mpfr::mpreal>>(init_eigs);
    Eigen::Vector<CppAD::AD<mpfr::mpreal>, Eigen::Dynamic> input_vector(num_inputs);
    input_vector = init_vector.reshaped();
    Eigen::Vector<CppAD::AD<mpfr::mpreal>, Eigen::Dynamic> output_vector(num_outputs);
    Eigen::Matrix<CppAD::AD<mpfr::mpreal>, Eigen::Dynamic, Eigen::Dynamic> work_matrix(n, n);
    // ----AD section begin----
    CppAD::Independent(input_vector);
    work_matrix = input_vector.reshaped(n, n);
    Eigen::SelfAdjointEigenSolver<decltype(work_matrix)> eig;
    eig.compute(work_matrix);
    output_vector = (eig.eigenvectors() * eig.eigenvalues().array().abs().log().matrix().asDiagonal() * eig.eigenvectors()).reshaped();
    CppAD::ADFun<mpfr::mpreal> f(input_vector, output_vector);
    // ----AD section end----
    Eigen::Vector<mpfr::mpreal, Eigen::Dynamic> final_eigs(n);
    final_eigs.setRandom();
    init_eigs = (2 * init_eigs.array() + 1).matrix();
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> final_vector(n, n);
    final_vector = randomPSD<mpfr::mpreal>(final_eigs);
    // Eigenf.Jacobian(v);
    Eigen::Vector<mpfr::mpreal, Eigen::Dynamic> v = final_vector.reshaped();
    Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> Mi = f.Jacobian(v).reshaped(num_outputs, num_inputs);
    // std::cout << "M:\n" << M << std::endl;
    // std::cout << "Calculated Jacobian:\n" << f.Jacobian(v) << std::endl;
}