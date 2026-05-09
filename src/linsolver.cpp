#include "linsolver.hpp"
#include "model.hpp"
#include "point.hpp"
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "common_typedefs.hpp"
#include <boost/range/irange.hpp>
#include <execution>

template<typename prec_type>
LinearSolver<prec_type>::LinearSolver(const Model<prec_type>& model, const Point<prec_type>& q){
    A = model.A.sparseView();
    G = model.G.sparseView();
    Gt = model.G.transpose().sparseView();
    At = model.A.transpose().sparseView();
    GtHG.resize(G.cols(), G.cols());
    rx.resize(model.n, 3); ry.resize(model.p, 3); rz.resize(model.d, 3);
        x.resize(model.n, 3);  y.resize(model.p, 3);  z.resize(model.d, 3);
    rx.col(1) = model.c; rx.col(2) = q.x;
    ry.col(1) = model.b; ry.col(2) = q.y; ry.col(0).setZero();
    rz.col(1) = model.h; rz.col(2) = q.z; rz.col(0).setZero();
    rx_minus_muGtHrz.resize(model.n, 3);
}

template<typename prec_type>
void LinearSolver<prec_type>::compute_aux_matrices(Model<prec_type>& model, const Point<prec_type>& q){
    // https://stackoverflow.com/questions/55845567/how-to-parallelize-a-plain-for-loop-using-the-c-standard-library
    auto ints = boost::irange<int>(0, model.G.cols());
    std::for_each_n(std::execution::par_unseq, ints.begin(), ints.size(), [&](int colIndex){
        GtHG(Eigen::placeholders::all, colIndex).noalias() = Gt * model.cone().hvp(model.G(Eigen::placeholders::all, colIndex));
    });
    lltG.compute(GtHG);
    lltA.compute(A * lltG.solve(model.A.transpose()));
    GtHrz1.noalias() = Gt * model.cone().hvp(rz.col(1));
    GtHrz2.noalias() = Gt * model.cone().hvp(rz.col(2));
    hvph = model.cone().hvp(model.h);
    hvpqz = model.cone().hvp(q.z);
}

template<typename prec_type>
Point<prec_type> LinearSolver<prec_type>::solve_ns(Model<prec_type>& model, const Point<prec_type>& p,
    const Point<prec_type>& q, prec_type mu, bool compute_aux){
    optVector<prec_type> pzg = p.z + mu * model.cone().jacobian();
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
    ABC = model.c.transpose() * x + model.b.transpose() * y - mu * hvph.transpose() * z;
    ABC(0) += - model.h.dot(pzg) + mu / p.tau - p.kap;
    ABC(1) += mu / (p.tau * p.tau);
    ABC(2) += q.tau;
    DEF = - q.x.transpose() * x - q.y.transpose() * y + mu * hvpqz.transpose() * z;
    DEF(0) += q.z.dot(pzg);
    DEF(1) += q.tau;
    // assemble 2 x 2 matrix and solve
    Eigen::Matrix<prec_type, 2, 2> mat{{ABC(1), ABC(2)}, {DEF(1), DEF(2)}};
    Eigen::Vector<prec_type, 2> con{ABC(0), DEF(0)};
    // Cholesky solve doesn't work here because mat is not positive-semidefinite
    // You cannot just throw Cholesky at everything
    // TODO: Are the previous instances of Cholesky (LLT) solve valid?
    Eigen::Vector<prec_type, 2> result = mat.inverse() * con;
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

template class LinearSolver<double>;