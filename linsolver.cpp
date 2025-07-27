#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include "point.cpp"
#include "model.cpp"

template<typename RealScalar>
class LinearSolver{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
private:
    Eigen::LLT<Matrix> llt;
    Point<RealScalar> d;
    Matrix At, Gt, I;
    PointXYS<RealScalar> b1, b2, b3, rhs;
public:
    LinearSolver(const Model<RealScalar>& model){
        Gt = model.G.transpose();
        At = model.A.transpose();
        I = Matrix::Identity(model.G.cols(), model.G.cols());
    }
    Point<RealScalar> solve_ns(Model<RealScalar>& model, const Point<RealScalar>& p, const Point<RealScalar>& q, RealScalar mu){
        // In this solver, we can modify it to use jacobian-vector product evaluations instead of computing the jacobian explicitly
        // But then you lose the ability to obtain the dual objective
        // We have the model (supplies the model data) and p (the current point)
        // this is the gradient
        Vector g = model.cone().jacobian();
        // first, I need to solve the 3 x 3 system to get x, y, s
        // I need (G H G^).inverse()
        // Maybe this whole process can be optimized...
        Matrix GtHG(model.G.cols(), model.G.cols());
        for(int rowIndex = 0; rowIndex < model.G.cols(); ++rowIndex){
            for(int colIndex = 0; colIndex < model.G.cols(); ++colIndex){
                GtHG(rowIndex, colIndex) = Gt.row(rowIndex) * model.cone().hvp(model.G.col(colIndex));
            }
        }
        // std::cout << "Determinant of GtHG = " << GtHG.determinant() << std::endl;
        Matrix GtHGinv = GtHG.llt().solve(I);
        Matrix AGtHGinv = model.A * GtHGinv;
        Matrix AGtHGinvAt = AGtHGinv * At;
        Vector bx_minus_muGtHbz;
        llt.compute(AGtHGinvAt);
        // Part 1: the constant part
        rhs.x = Gt * (p.z + mu * g);
        b1.y = llt.solve(AGtHGinv * rhs.x);
        b1.x = GtHGinv * (rhs.x - At * b1.y) / mu;
        b1.s = - model.G * b1.x;
        // Part 2: the part proportional to dtau
        rhs.x = model.c;
        rhs.y = model.b;
        rhs.s = model.h;
        bx_minus_muGtHbz = rhs.x - mu * Gt * model.cone().hvp(rhs.s);
        b2.y = llt.solve(AGtHGinv * bx_minus_muGtHbz + mu * rhs.y);
        b2.x = GtHGinv * (bx_minus_muGtHbz - At * b2.y) / mu;
        b2.s = -(rhs.s + model.G * b2.x);
        // Part 3: the part proportional to dtheta
        rhs.x = q.x;
        rhs.y = q.y;
        rhs.s = q.z;
        bx_minus_muGtHbz = rhs.x - mu * Gt * model.cone().hvp(rhs.s);
        b3.y = llt.solve(AGtHGinv * bx_minus_muGtHbz + mu * rhs.y);
        b3.x = GtHGinv * (bx_minus_muGtHbz - At * b3.y) / mu;
        b3.s = -(rhs.s + model.G * b3.x);
        // now we solve for dtau and dtheta
        RealScalar A, B, C, D, E, F;
        Vector hvph = model.cone().hvp(model.h);
        Vector hvpqz = model.cone().hvp(q.z);
        A = - model.c.dot(b1.x) - model.b.dot(b1.y) + mu * hvph.dot(b1.s) + model.h.dot(p.z + mu * g) - mu / p.tau + p.kap;
        B = model.c.dot(b2.x) + model.b.dot(b2.y) - mu * hvph.dot(b2.s) + mu / (p.tau * p.tau);
        C = model.c.dot(b3.x) + model.b.dot(b3.y) - mu * hvph.dot(b3.s) + q.tau;
        D = q.x.dot(b1.x) + q.y.dot(b1.y) - mu * hvpqz.dot(b1.s) - q.z.dot(p.z + mu * g);
        E = - q.x.dot(b2.x) - q.y.dot(b2.y) + mu * hvpqz.dot(b2.s) + q.tau;
        F = - q.x.dot(b3.x) - q.y.dot(b3.y) + mu * hvpqz.dot(b3.s);
        // assemble 2 x 2 matrix and solve
        Eigen::Matrix<RealScalar, 2, 2> mat;
        Eigen::Vector<RealScalar, 2> con;
        mat << B, C, E, F;
        con << -A, -D;
        // Cholesky solve doesn't work here because mat is not positive-semidefinite
        // You cannot just throw Cholesky at everything
        // TODO: Are the previous instances of Cholesky (LLT) solve valid?
        Eigen::Vector<RealScalar, 2> result = mat.colPivHouseholderQr().solve(con);
        d.tau = result(0, 0);
        d.theta = result(1, 0);
        // can set dx, dy, ds now
        d.x = b1.x - d.tau * b2.x - d.theta * b3.x;
        d.y = b1.y - d.tau * b2.y - d.theta * b3.y;
        d.s = b1.s - d.tau * b2.s - d.theta * b3.s;
        // can set dz and dkap now
        d.z = - p.z - mu * g - mu * model.cone().hvp(d.s);
        d.kap = - p.kap + mu / p.tau - mu * d.tau / (p.tau * p.tau);
        return d;
    }
};

#endif