#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "common_typedefs.hpp"
#include "problem_data.hpp"

/// @brief Class that solves the Newton system of the ESDE.
/// @tparam prec_type Underlying precision type of the solver.
template<typename prec_type>
class ESDENewtonSolver{
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> rx, ry, rz;
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> x, y, z;
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> rx_minus_muGtHrz;
    Eigen::LDLT<optMatrix<prec_type>> lltA, lltG;
    Point<prec_type> d;
    Eigen::SparseMatrix<prec_type> A, G, At, Gt;
    optMatrix<prec_type> HG;
    optVector<prec_type> GtHrz1, GtHrz2, hvph, hvpqz;
    Eigen::RowVector<prec_type, 3> ABC, DEF;
public:
    LinearSolver(const Model<prec_type>& model, const Point<prec_type>& q);
    void compute_aux_matrices(Model<prec_type>& model, const Point<prec_type>& q);
    Point<prec_type> solve_ns(Model<prec_type>& model, const Point<prec_type>& p,
        const Point<prec_type>& q, prec_type mu, bool compute_aux=true);
};

/// @brief Solves the ESDE Newton system given a right hand side.
/// @tparam prec_type Underlying precision type of the solver.
/// @param problem_data Class that holds the problem data.
/// @param rhs The right hand side to solve using.
/// @return The solution of the ESDE Newton system.
template<typename prec_type>
optVector<prec_type> esde_newton_system_solver(const ProblemData<prec_type>& problem_data, optVector<prec_type> rhs){
    // TODO: The second argument is a pass-by-value, so check if that can be optimized
    // first, we need to extract the subvectors corresponding to x, y, z, tau, s, kappa
    auto rx = rhs(Eigen::seqN(0, problem_data.A.cols()));
    auto ry = rhs(Eigen::seqN(rx.size(), problem_data.A.rows()));
    auto rz = rhs(Eigen::seqN(rx.size() + ry.size(), problem_data.G.rows()));
    auto rtau = rhs(rx.size() + ry.size() + rz.size());
    auto rs = rhs(Eigen::seqN(rx.size() + ry.size() + rz.size() + 1, problem_data.G.rows()));
    auto rkap = rhs(Eigen::placeholders::last);
    
}

#endif