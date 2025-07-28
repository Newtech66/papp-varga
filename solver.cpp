#ifndef SOLVER_PAPP_VARGA_H
#define SOLVER_PAPP_VARGA_H
#include <string>
#include <Eigen/Core>
#include <mpreal.h>
#include "model.cpp"
#include "point.cpp"
#include "linsolver.cpp"

template<typename RealScalar>
class Solver{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
private:
    Point<RealScalar> p, dp, q;
    RealScalar mu, nu;
    int steps_taken;
public:
    Point<RealScalar> solve(Model<RealScalar>& model, const RealScalar& tol_gap, const RealScalar& tol_fail, const int max_steps = 1000);
    void set_init_point(Model<RealScalar>& model);
    RealScalar calc_nu(Model<RealScalar>& model);
    RealScalar calc_iterate_norm(Model<RealScalar>& model, const RealScalar& mu);
    Vector compute_newton_residuals(Model<RealScalar>& model, const Point<RealScalar>& d) const;
    void geometric_update_mu(Model<RealScalar>& model);
    void adaptive_update_mu(Model<RealScalar>& model);
};

template<typename RealScalar>
RealScalar Solver<RealScalar>::calc_nu(Model<RealScalar>& model){
    int barrier_parameter = 1 + model.cone().barrierParameter();
    return 1 / (8 * (1 + std::sqrt(barrier_parameter)));
}

template<>
mpfr::mpreal Solver<mpfr::mpreal>::calc_nu(Model<mpfr::mpreal>& model){
    int barrier_parameter = 1 + model.cone().barrierParameter();
    return mpfr::mpreal("0.125") / (mpfr::mpreal(1) + mpfr::sqrt(barrier_parameter));
}

template<typename RealScalar>
RealScalar Solver<RealScalar>::calc_iterate_norm(Model<RealScalar>& model, const RealScalar& mu){
    // ||z + mu * g(s) + tau + mu * g(kap)||*_(s, kap)
    // kap and tau are in non-neg orthant (>=0)
    // the barrier function is -log(x), gradient is -1/x, hessian is 1/x^2
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    Vector t = p.z + mu * model.cone().jacobian();
    // sqrt(t^ * H^-1 * t)
    RealScalar n2 = t.dot(model.cone().ihvp(t));
    // add the kap tau part
    // (tau - mu / kap) * kap^2 (tau - mu / kap) = (kap * tau - mu) ** 2
    n2 += (p.kap * p.tau - mu) * (p.kap * p.tau - mu);
    return std::sqrt(n2);
}

template<>
mpfr::mpreal Solver<mpfr::mpreal>::calc_iterate_norm(Model<mpfr::mpreal>& model, const mpfr::mpreal& mu){
    using Vector = Eigen::Vector<mpfr::mpreal, Eigen::Dynamic>;
    Vector t = p.z + mu * model.cone().jacobian();
    mpfr::mpreal n2 = t.dot(model.cone().ihvp(t));
    n2 += (p.kap * p.tau - mu) * (p.kap * p.tau - mu);
    return mpfr::sqrt(n2);
}

template<typename RealScalar>
Eigen::Vector<RealScalar, Eigen::Dynamic> Solver<RealScalar>::compute_newton_residuals(Model<RealScalar>& model, const Point<RealScalar>& d) const{
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    Matrix T = Matrix::Zero(model.n + model.p + model.d + 2, model.n + model.p + model.d + 2);
    T.block(0, model.n, model.n, model.p) = model.A.transpose();
    T.block(0, model.n + model.p, model.n, model.d) = model.G.transpose();
    T.block(0, model.n + model.p + model.d, model.n, 1) = model.c;
    T.block(0, model.n + model.p + model.d + 1, model.n, 1) = q.x;
    T.block(model.n, model.n + model.p + model.d, model.p, 1) = model.b;
    T.block(model.n, model.n + model.p + model.d + 1, model.p, 1) = q.y;
    T.block(model.n + model.p, model.n + model.p + model.d, model.d, 1) = model.h;
    T.block(model.n + model.p, model.n + model.p + model.d + 1, model.d, 1) = q.z;
    T(model.n + model.p + model.d, model.n + model.p + model.d + 1) = q.tau;
    // Make skew symmetric
    T -= T.transpose().eval();
    Vector A = Vector::Zero(T.cols());
    Vector B = Vector::Zero(T.rows());
    A.segment(0, model.n) = d.x;
    A.segment(model.n, model.p) = d.y;
    A.segment(model.n + model.p, model.d) = d.z;
    A(model.n + model.p + model.d) = d.tau;
    A(model.n + model.p + model.d + 1) = d.theta;
    B.segment(model.n + model.p, model.d) = d.s;
    B(model.n + model.p + model.d) = d.kap;
    return T * A - B;
}

template<typename RealScalar>
Point<RealScalar> Solver<RealScalar>::solve(Model<RealScalar>& model, const RealScalar& tol_gap, const RealScalar& tol_fail, const int max_steps){
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    set_init_point(model);
    mu = RealScalar(1);
    LinearSolver solver(model);
    steps_taken = 0;
    while(p.s.dot(p.z) + p.tau * p.kap > tol_gap){
        if(steps_taken >= max_steps){
            std::cout << "Exiting because max iterations were reached" << std::endl;
            break;
        }
        Point<RealScalar> d = solver.solve_ns(model, p, q, mu);
        p += d;
        model.cone().updatePoint(p.s);
        // geometric_update_mu(model);
        adaptive_update_mu(model);
        // tau should be non-neg
        if(p.tau < RealScalar(0)){
            std::cout << "Exiting because tau became negative" << std::endl;
            p -= d;
            break;
        }
        // kap should be non-neg
        if(p.kap < RealScalar(0)){
            std::cout << "Exiting because kap became negative" << std::endl;
            p -= d;
            break;
        }
        ++steps_taken;
    }
    std::cout << "Iterations taken = " << steps_taken << std::endl;
    std::cout << "Solver status: ";
    if(p.tau < tol_fail and p.kap < tol_fail)   std::cout << "UNKNOWN (tau = 0, kap = 0)";
    else if(p.kap > tol_fail and p.tau > tol_fail)  std::cout << "UNKNOWN (tau > 0, kap > 0)";
    else if(p.kap > tol_fail){
        bool pinfeas = model.h.dot(p.z) + model.b.dot(p.y) < RealScalar(0);
        bool dinfeas = model.c.dot(p.x) < RealScalar(0);
        if(pinfeas and dinfeas) std::cout << "PRIMAL_DUAL_INFEASIBLE";
        else if(pinfeas) std::cout << "PRIMAL_INFEASIBLE";
        else if(dinfeas) std::cout << "DUAL_INFEASIBLE";
        else    std::cout << "INFEASIBLE";
    }else if(p.tau > tol_fail)  std::cout << "OPTIMAL";
    else   std::cout << "UNEXPECTED";
    std::cout << std::endl;
    return p;
}

template<typename RealScalar>
void Solver<RealScalar>::geometric_update_mu(Model<RealScalar>& model){
    nu = calc_nu(model);
    mu *= RealScalar(1) - nu;
}

template<typename RealScalar>
void Solver<RealScalar>::adaptive_update_mu(Model<RealScalar>& model){
    // ||z + tau||*_(s, kap)
    // sqrt(z^ * H^-1 * z)
    RealScalar n2 = p.z.dot(model.cone().ihvp(p.z));
    // add the kap tau part
    // tau * kap^2 * tau = (kap * tau) ** 2
    n2 += (p.kap * p.tau) * (p.kap * p.tau);
    int barrier_parameter = 1 + model.cone().barrierParameter();
    RealScalar a = RealScalar(barrier_parameter) - RealScalar(1) / RealScalar(16);
    RealScalar b = p.z.dot(p.s) + p.tau * p.kap;
    mu = (b - std::sqrt(b * b - a * n2)) / a;
}

template<>
void Solver<mpfr::mpreal>::adaptive_update_mu(Model<mpfr::mpreal>& model){
    // ||z + tau||*_(s, kap)
    // sqrt(z^ * H^-1 * z)
    mpfr::mpreal n2 = p.z.dot(model.cone().ihvp(p.z));
    // add the kap tau part
    // tau * kap^2 * tau = (kap * tau) ** 2
    n2 += (p.kap * p.tau) * (p.kap * p.tau);
    int barrier_parameter = 1 + model.cone().barrierParameter();
    mpfr::mpreal a = mpfr::mpreal(barrier_parameter) - mpfr::mpreal(1) / mpfr::mpreal(16);
    mpfr::mpreal b = p.z.dot(p.s) + p.tau * p.kap;
    mu = (b - mpfr::sqrt(b * b - a * n2)) / a;
}

template<typename RealScalar>
void Solver<RealScalar>::set_init_point(Model<RealScalar>& model){
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    // x = 0, y = 0, z = -g(s), s = (set to an interior point of the cone)
    // kap = tau = theta = 1
    // x -> R[n, 1], y -> R[p, 1], z -> R[d, 1], s -> R[d, 1]
    p.x = Vector::Zero(model.n);
    p.y = Vector::Zero(model.p);
    p.s = model.cone().point();
    p.z = -model.cone().jacobian();
    p.kap = p.tau = p.theta = RealScalar(1);
    // calculate q
    q.x = - model.A.transpose() * p.y - model.G.transpose() * p.z - model.c;
    q.y = model.A * p.x - model.b;
    q.z = model.G * p.x - model.h + p.s;
    q.tau = model.c.dot(p.x) + model.b.dot(p.y) + model.h.dot(p.z) + RealScalar(1);
}

#endif