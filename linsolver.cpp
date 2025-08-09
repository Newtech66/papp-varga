#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include "model.cpp"
#include "point.cpp"

template<typename RealScalar>
class LinearSolver{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::LDLT<Matrix> lltA, lltG;
    Point<RealScalar> d;
    Matrix At, Gt;
public:
    LinearSolver(const Model<RealScalar>& model){
        Gt = model.G.transpose();
        At = model.A.transpose();
    }
    Point<RealScalar> solve_ns(Model<RealScalar>& model, const Point<RealScalar>& p, const Point<RealScalar>& q, RealScalar mu){
        Vector pzg = p.z + mu * model.cone().jacobian();
        Matrix GtHG(Gt.rows(), Gt.rows());
        for(int colIndex = 0; colIndex < model.G.cols(); ++colIndex){
            GtHG.col(colIndex).noalias() = Gt * model.cone().hvp(model.G.col(colIndex));
        }
        lltG.compute(GtHG);
        Matrix AGtHGinvAt(model.A.rows(), model.A.rows());
        AGtHGinvAt.noalias() = model.A * lltG.solve(At);
        lltA.compute(AGtHGinvAt);
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx(model.n, 3), ry(model.p, 3), rz(model.d, 3);
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> x(model.n, 3), y(model.p, 3), z(model.d, 3);
        Eigen::Matrix<RealScalar, Eigen::Dynamic, 3> rx_minus_muGtHrz(model.n, 3);
        rx.col(1) = model.c; rx.col(2) = q.x; rx.col(0).noalias() = Gt * pzg;
        ry.col(1) = model.b; ry.col(2) = q.y; ry.col(0).setZero();
        rz.col(1) = model.h; rz.col(2) = q.z; rz.col(0).setZero();
        // Step 1: Calculate rx - mu * Gt * H * rz
        rx_minus_muGtHrz = rx;
        rx_minus_muGtHrz.col(1).noalias() -= mu * Gt * model.cone().hvp(rz.col(1));
        rx_minus_muGtHrz.col(2).noalias() -= mu * Gt * model.cone().hvp(rz.col(2));
        // Step 2: Calculate y
        y = mu * ry;
        y.noalias() += model.A * lltG.solve(rx_minus_muGtHrz);
        y = lltA.solve(y);
        // Step 3: Calculate x
        x = rx_minus_muGtHrz;
        x.noalias() -= At * y;
        x = lltG.solve(x / mu);
        // Step 4: Calculate z
        z = -rz;
        z.noalias() -= model.G * x;
        // now we solve for dtau and dtheta
        RealScalar A, B, C, D, E, F;
        Vector hvph = model.cone().hvp(model.h);
        Vector hvpqz = model.cone().hvp(q.z);
        A = model.c.dot(x.col(0)) + model.b.dot(y.col(0)) - mu * hvph.dot(z.col(0)) - model.h.dot(pzg) + mu / p.tau - p.kap;
        B = model.c.dot(x.col(1)) + model.b.dot(y.col(1)) - mu * hvph.dot(z.col(1)) + mu / (p.tau * p.tau);
        C = model.c.dot(x.col(2)) + model.b.dot(y.col(2)) - mu * hvph.dot(z.col(2)) + q.tau;
        D = - q.x.dot(x.col(0)) - q.y.dot(y.col(0)) + mu * hvpqz.dot(z.col(0)) + q.z.dot(pzg);
        E = - q.x.dot(x.col(1)) - q.y.dot(y.col(1)) + mu * hvpqz.dot(z.col(1)) + q.tau;
        F = - q.x.dot(x.col(2)) - q.y.dot(y.col(2)) + mu * hvpqz.dot(z.col(2));
        // assemble 2 x 2 matrix and solve
        Eigen::Matrix<RealScalar, 2, 2> mat{{B, C}, {E, F}};
        Eigen::Vector<RealScalar, 2> con{A, D};
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