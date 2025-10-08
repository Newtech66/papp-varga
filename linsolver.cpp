#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include <Eigen/Cholesky>
#include "model.cpp"
#include "point.cpp"
#include <Eigen/SparseCore>

template<typename RealScalar>
class LinearSolver{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx, ry, rz;
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> x, y, z;
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx_minus_muGtHrz;
    Eigen::LDLT<Matrix> lltA, lltG;
    Point<RealScalar> d;
    Eigen::SparseMatrix<RealScalar> A, G, At, Gt;
    Matrix GtHG;
    Vector GtHrz1, GtHrz2, hvph, hvpqz;
public:
    LinearSolver(const Model<RealScalar>& model, const Point<RealScalar>& q){
        A = model.A.sparseView();
        G = model.G.sparseView();
        Gt = model.G.transpose().sparseView();
        At = model.A.transpose().sparseView();
        GtHG.resize(Gt.rows(), Gt.rows());
        rx.resize(model.n, 3); ry.resize(model.p, 3); rz.resize(model.d, 3);
         x.resize(model.n, 3);  y.resize(model.p, 3);  z.resize(model.d, 3);
        rx.col(1) = model.c; rx.col(2) = q.x;
        ry.col(1) = model.b; ry.col(2) = q.y; ry.col(0).setZero();
        rz.col(1) = model.h; rz.col(2) = q.z; rz.col(0).setZero();
        rx_minus_muGtHrz.resize(model.n, 3);
    }
    void compute_aux_matrices(Model<RealScalar>& model, const Point<RealScalar>& q){
        for(int colIndex = 0; colIndex < model.G.cols(); ++colIndex){
            GtHG.col(colIndex).noalias() = Gt * model.cone().hvp(model.G.col(colIndex));
        }
        lltG.compute(GtHG);
        lltA.compute(A * lltG.solve(model.A.transpose()));
        GtHrz1.noalias() = Gt * model.cone().hvp(rz.col(1));
        GtHrz2.noalias() = Gt * model.cone().hvp(rz.col(2));
        hvph = model.cone().hvp(model.h);
        hvpqz = model.cone().hvp(q.z);
    }
    Point<RealScalar> solve_ns(Model<RealScalar>& model, const Point<RealScalar>& p, const Point<RealScalar>& q, RealScalar mu, bool compute_aux=true){
        Vector pzg = p.z + mu * model.cone().jacobian();
        rx.col(0).noalias() = Gt * pzg;
        if(compute_aux) compute_aux_matrices(model, q);
        // Step 1: Calculate rx - mu * Gt * H * rz
        rx_minus_muGtHrz = rx;
        rx_minus_muGtHrz.col(1).noalias() -= mu * GtHrz1;
        rx_minus_muGtHrz.col(2).noalias() -= mu * GtHrz2;
        // Step 2: Calculate y
        y = mu * ry;
        y.noalias() += A * lltG.solve(rx_minus_muGtHrz);
        y = lltA.solve(y);
        // Step 3: Calculate x
        x = rx_minus_muGtHrz;
        x.noalias() -= At * y;
        x = lltG.solve(x / mu);
        // Step 4: Calculate z
        z = -rz;
        z.noalias() -= G * x;
        // now we solve for dtau and dtheta
        Eigen::RowVector<RealScalar, 3> ABC, DEF;
        ABC = model.c.transpose() * x + model.b.transpose() * y - mu * hvph.transpose() * z;
        ABC(0) += - model.h.dot(pzg) + mu / p.tau - p.kap;
        ABC(1) += mu / (p.tau * p.tau);
        ABC(2) += q.tau;
        DEF = - q.x.transpose() * x - q.y.transpose() * y + mu * hvpqz.transpose() * z;
        DEF(0) += q.z.dot(pzg);
        DEF(1) += q.tau;
        // assemble 2 x 2 matrix and solve
        Eigen::Matrix<RealScalar, 2, 2> mat{{ABC(1), ABC(2)}, {DEF(1), DEF(2)}};
        Eigen::Vector<RealScalar, 2> con{ABC(0), DEF(0)};
        // Cholesky solve doesn't work here because mat is not positive-semidefinite
        // You cannot just throw Cholesky at everything
        // TODO: Are the previous instances of Cholesky (LLT) solve valid?
        Eigen::Vector<RealScalar, 2> result = mat.inverse() * con;
        d.tau = result(0);
        d.theta = result(1);
        // can set dx, dy, ds now
        d.x = x.col(0) - d.tau * x.col(1) - d.theta * x.col(2);
        d.y = y.col(0) - d.tau * y.col(1) - d.theta * y.col(2);
        d.s = z.col(0) - d.tau * z.col(1) - d.theta * z.col(2);
        // can set dz and dkap now
        d.z = - pzg - mu * model.cone().hvp(d.s);
        d.kap = - p.kap + mu / p.tau - mu * d.tau / (p.tau * p.tau);
        return d;
    }
};

#endif