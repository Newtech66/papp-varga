#ifndef NESTEROV_TODD_SOLVERS_H
#define NESTEROV_TODD_SOLVERS_H
#include "esde_state.hpp"
#include <memory>   
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

template<typename prec_type>
class NesterovToddSolver{
private:
    optMatrix<prec_type> scaling_matrix; // this is the scaling matrix
    optVector<prec_type> lambda; // this is lambda = W * z = (W^-1)^T * s
    optVector<prec_type> scaling_point; // this is the scaling point
    const ESDEState<prec_type>& esde_state; // the state of the ESDE

    Eigen::SelfAdjointEigenSolver<prec_type> sym_eigsolver;
    Eigen::SelfAdjointEigenSolver<std::complex<prec_type>> her_eigsolver;

    Eigen::BDCSVD<prec_type, Eigen::ComputeThinV> sym_svd;
    Eigen::BDCSVD<std::complex<prec_type>, Eigen::ComputeThinV> her_svd;

    optVector<prec_type> findPredictionDirection();
    prec_type findCenteringParameter();
    prec_type findLargestStepSize();
    optVector<prec_type> findCombinedDirection();

public:
    NesterovToddSolver(const ESDEState<prec_type>&);
    void updateInternalState();
    void step();
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&);
};

#endif