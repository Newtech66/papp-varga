#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include "model.cpp"
#include "point.cpp"
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "common_typedefs.hpp"

template<typename RealScalar>
class LinearSolver{
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx, ry, rz;
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> x, y, z;
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx_minus_muGtHrz;
    Eigen::LDLT<optMatrix<RealScalar>> lltA, lltG;
    Point<RealScalar> d;
    Eigen::SparseMatrix<RealScalar> A, G, At, Gt;
    optMatrix<RealScalar> GtHG;
    optVector<RealScalar> GtHrz1, GtHrz2, hvph, hvpqz;
public:
    LinearSolver(const Model<RealScalar>& model, const Point<RealScalar>& q);
    void compute_aux_matrices(Model<RealScalar>& model, const Point<RealScalar>& q);
    Point<RealScalar> solve_ns(Model<RealScalar>& model, const Point<RealScalar>& p,
        const Point<RealScalar>& q, RealScalar mu, bool compute_aux=true);
};

#endif