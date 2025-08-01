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
    Eigen::LLT<Matrix> lltA;
    Point<RealScalar> d;
    Matrix At, Gt, IG, IA;
    PointXYS<RealScalar> b1, b2, b3;
public:
    LinearSolver(const Model<RealScalar>& model){
        Gt = model.G.transpose();
        At = model.A.transpose();
        IG = Matrix::Identity(Gt.rows(), Gt.rows());
        IA = Matrix::Identity(At.cols(), At.cols());
    }
    Point<RealScalar> solve_ns(Model<RealScalar>& model, const Point<RealScalar>& p, const Point<RealScalar>& q, RealScalar mu){
        // In this solver, we can modify it to use jacobian-vector product evaluations instead of computing the jacobian explicitly
        // But then you lose the ability to obtain the dual objective
        // We have the model (supplies the model data) and p (the current point)
        // this is the p.z + mu * g
        Vector pzg = p.z + mu * model.cone().jacobian();
        // first, I need to solve the 3 x 3 system to get x, y, s
        // I need (G H G^).inverse()
        Matrix GtHG(Gt.rows(), Gt.rows());
        for(int colIndex = 0; colIndex < model.G.cols(); ++colIndex){
            GtHG.col(colIndex).noalias() = Gt * model.cone().hvp(model.G.col(colIndex));
        }
        Matrix GtHGinv = GtHG.llt().solve(IG);
        Matrix AGtHGinv(model.A.rows(), Gt.rows());
        AGtHGinv.noalias() = model.A * GtHGinv;
        Matrix AGtHGinvAt(model.A.rows(), model.A.rows());
        AGtHGinvAt.noalias() = AGtHGinv * At;
        lltA.compute(AGtHGinvAt);
        // Matrix AGtHGinvAtinv = AGtHGinvAt.llt().solve(IA);
        // std::cout << "GtHG =" << std::endl;
        // std::cout << GtHG << std::endl;
        // std::cout << "(GtHG)^-1 =" << std::endl;
        // std::cout << GtHGinv << std::endl;
        // std::cout << "A(GtHG)^-1 =" << std::endl;
        // std::cout << AGtHGinv << std::endl;
        // std::cout << "A(GtHG)^-1At =" << std::endl;
        // std::cout << AGtHGinvAt << std::endl;
        // std::cout << "(A(GtHG)^-1At)^-1 =" << std::endl;
        // std::cout << AGtHGinvAtinv << std::endl;
        Vector bx_minus_muGtHbz;
        // Part 1: the constant part
        Vector rhsx(Gt.rows());
        rhsx.noalias() = Gt * pzg;
        // b1.y.noalias() = AGtHGinvAtinv * AGtHGinv * rhsx;
        b1.y.noalias() = lltA.solve(AGtHGinv * rhsx);
        b1.x.noalias() = GtHGinv * rhsx / mu;
        b1.x.noalias() -= GtHGinv * At * b1.y / mu;
        b1.s.noalias() = - model.G * b1.x;
        // Part 2: the part proportional to dtau
        bx_minus_muGtHbz = model.c;
        bx_minus_muGtHbz.noalias() -= mu * Gt * model.cone().hvp(model.h);
        b2.y.noalias() = AGtHGinv * bx_minus_muGtHbz;
        b2.y += mu * model.b;
        b2.y = lltA.solve(b2.y);
        b2.x.noalias() = GtHGinv * bx_minus_muGtHbz / mu;
        b2.x.noalias() -= GtHGinv * At * b2.y / mu;
        b2.s = -model.h;
        b2.s.noalias() -= model.G * b2.x;
        // Part 3: the part proportional to dtheta
        bx_minus_muGtHbz = q.x;
        bx_minus_muGtHbz.noalias() -= mu * Gt * model.cone().hvp(q.z);
        b3.y.noalias() = AGtHGinv * bx_minus_muGtHbz;
        b3.y += mu * q.y;
        b3.y = lltA.solve(b3.y);
        b3.x.noalias() = GtHGinv * bx_minus_muGtHbz / mu;
        b3.x.noalias() -= GtHGinv * At * b3.y / mu;
        b3.s = -q.z;
        b3.s.noalias() -= model.G * b3.x;
        // now we solve for dtau and dtheta
        RealScalar A, B, C, D, E, F;
        Vector hvph = model.cone().hvp(model.h);
        Vector hvpqz = model.cone().hvp(q.z);
        A = - model.c.dot(b1.x) - model.b.dot(b1.y) + mu * hvph.dot(b1.s) + model.h.dot(pzg) - mu / p.tau + p.kap;
        B = model.c.dot(b2.x) + model.b.dot(b2.y) - mu * hvph.dot(b2.s) + mu / (p.tau * p.tau);
        C = model.c.dot(b3.x) + model.b.dot(b3.y) - mu * hvph.dot(b3.s) + q.tau;
        D = q.x.dot(b1.x) + q.y.dot(b1.y) - mu * hvpqz.dot(b1.s) - q.z.dot(pzg);
        E = - q.x.dot(b2.x) - q.y.dot(b2.y) + mu * hvpqz.dot(b2.s) + q.tau;
        F = - q.x.dot(b3.x) - q.y.dot(b3.y) + mu * hvpqz.dot(b3.s);
        // assemble 2 x 2 matrix and solve
        Eigen::Matrix<RealScalar, 2, 2> mat{{B, C}, {E, F}};
        Eigen::Vector<RealScalar, 2> con{-A, -D};
        // Cholesky solve doesn't work here because mat is not positive-semidefinite
        // You cannot just throw Cholesky at everything
        // TODO: Are the previous instances of Cholesky (LLT) solve valid?
        // std::cout << "mat =" << std::endl;
        // std::cout << mat << std::endl;
        // std::cout << "mat^-1 =" << std::endl;
        // std::cout << mat.inverse() << std::endl;
        // std::cout << "con =" << std::endl;
        // std::cout << con << std::endl;
        Eigen::Vector<RealScalar, 2> result = mat.inverse() * con;
        d.tau = result(0);
        d.theta = result(1);
        // can set dx, dy, ds now
        d.x = b1.x - d.tau * b2.x - d.theta * b3.x;
        d.y = b1.y - d.tau * b2.y - d.theta * b3.y;
        d.s = b1.s - d.tau * b2.s - d.theta * b3.s;
        // can set dz and dkap now
        d.z = - pzg - mu * model.cone().hvp(d.s);
        d.kap = - p.kap + mu / p.tau - mu * d.tau / (p.tau * p.tau);
        return d;
    }
};

#endif