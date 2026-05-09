#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include "model.hpp"
#include "point.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "common_typedefs.hpp"

template<typename prec_type>
class LinearSolver{
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> rx, ry, rz;
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> x, y, z;
    Eigen::Matrix<prec_type, Eigen::Dynamic, 3> rx_minus_muGtHrz;
    Eigen::LDLT<optMatrix<prec_type>> lltA, lltG;
    Point<prec_type> d;
    Eigen::SparseMatrix<prec_type> A, G, At, Gt;
    optMatrix<prec_type> GtHG;
    optVector<prec_type> GtHrz1, GtHrz2, hvph, hvpqz;
    Eigen::RowVector<prec_type, 3> ABC, DEF;
public:
    LinearSolver(const Model<prec_type>& model, const Point<prec_type>& q);
    void compute_aux_matrices(Model<prec_type>& model, const Point<prec_type>& q);
    Point<prec_type> solve_ns(Model<prec_type>& model, const Point<prec_type>& p,
        const Point<prec_type>& q, prec_type mu, bool compute_aux=true);
};

#endif