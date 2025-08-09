#ifndef SOLVER_PAPP_VARGA_H
#define SOLVER_PAPP_VARGA_H
#include <string>
#include <chrono>
#include <thread>
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
    std::vector<RealScalar> alpha_sched;
    void set_init_point(Model<RealScalar>& model);
    void adaptive_update_mu(Model<RealScalar>& model);
    void print_header() const;
    void print_row(Model<RealScalar>& model, int, const std::chrono::duration<double, std::milli>&) const;
    // unused
    void calc_nu(Model<RealScalar>& model);
    RealScalar calc_iterate_norm(Model<RealScalar>& model, const RealScalar& mu);
    Vector compute_newton_residuals(Model<RealScalar>& model) const;
public:
    Point<RealScalar> solve(Model<RealScalar>& model, const RealScalar& tol_gap, const RealScalar& tol_fail, const int max_steps = 1000);
};

template<typename RealScalar>
void Solver<RealScalar>::print_header() const{
    std::cout << std::left << std::setw(8)  << "Step";
    std::cout << std::left << std::setw(16) << "Primal";
    std::cout << std::left << std::setw(16) << "Dual";
    std::cout << std::left << std::setw(14) << "tau";
    std::cout << std::left << std::setw(14) << "kap";
    std::cout << std::left << std::setw(14) << "mu";
    std::cout << std::left << std::setw(8)  << "istep";
    std::cout << std::left << std::setw(10) << "time (ms)";
    std::cout << std::endl;
}
template<typename RealScalar>
void Solver<RealScalar>::print_row(Model<RealScalar>& model, int istep, const std::chrono::duration<double, std::milli>& step_time) const{
    std::cout << std::left << std::setw(8)  << steps_taken;
    std::cout << std::left << std::setw(16) << std::scientific << std::setprecision(6) << model.c.dot(p.x) / p.tau;
    std::cout << std::left << std::setw(16) << std::scientific << std::setprecision(6) << (- model.h.dot(p.z) - model.b.dot(p.y)) / p.tau;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << p.tau;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << p.kap;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << mu;
    std::cout << std::left << std::setw(8)  << istep;
    std::cout << std::left << std::setw(10) << std::fixed << std::setprecision(2) << step_time.count();
    std::cout << std::endl;
}

template<typename RealScalar>
Point<RealScalar> Solver<RealScalar>::solve(Model<RealScalar>& model, const RealScalar& tol_gap, const RealScalar& tol_fail, const int max_steps){
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    set_init_point(model);
    mu = RealScalar(1);
    calc_nu(model);
    LinearSolver solver(model);
    steps_taken = 0;
    print_header();
    while(p.s.dot(p.z) + p.tau * p.kap > tol_gap){
        if(steps_taken >= max_steps){
            std::cout << "Exiting because max iterations were reached" << std::endl;
            break;
        }
        // adaptive update
        // auto solve_start = std::chrono::high_resolution_clock::now();
        // dp = solver.solve_ns(model, p, q, mu);
        // p += dp;
        // model.cone().updatePoint(p.s);
        // adaptive_update_mu(model);
        // auto solve_end = std::chrono::high_resolution_clock::now();
        //largest update
        auto solve_start = std::chrono::high_resolution_clock::now();
        int istep = 0;
        mu *= RealScalar(1) - nu;
        while(istep < alpha_sched.size()){
            RealScalar nextmu = mu * alpha_sched[istep];
            dp = solver.solve_ns(model, p, q, nextmu);
            p += dp;
            model.cone().updatePoint(p.s);
            // check in neighbourhood
            if(calc_iterate_norm(model, nextmu) > nextmu / RealScalar(4)){
                p -= dp;
                model.cone().updatePoint(p.s);
                break;
            }
            p -= dp;
            model.cone().updatePoint(p.s);
            ++istep;
        }
        mu *= alpha_sched[istep - 1];
        dp = solver.solve_ns(model, p, q, mu);
        p += dp;
        model.cone().updatePoint(p.s);
        // int istep = 0;
        // RealScalar lo = 0.9, hi = 1.0, mid, res = 1.0;
        // mu *= RealScalar(1) - nu;
        // while(istep < 10){
        //     mid = (lo + hi) / RealScalar(2);
        //     dp = solver.solve_ns(model, p, q, mu * mid);
        //     p += dp;
        //     model.cone().updatePoint(p.s);
        //     // check in neighbourhood
        //     if(mu * mid / RealScalar(4) < calc_iterate_norm(model, mu * mid))   lo = mid;
        //     else   hi = res = mid;
        //     p -= dp;
        //     model.cone().updatePoint(p.s);
        //     ++istep;
        // }
        // mu *= res;
        // dp = solver.solve_ns(model, p, q, mu);
        // p += dp;
        // model.cone().updatePoint(p.s);
        auto solve_end = std::chrono::high_resolution_clock::now();
        ++steps_taken;
        print_row(model, istep, solve_end - solve_start);
    }
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Iterations taken = " << steps_taken << std::endl;
    // std::cout << "Solver status: ";
    // if(p.tau < tol_fail and p.kap < tol_fail)   std::cout << "UNKNOWN (tau = 0, kap = 0)";
    // else if(p.kap > tol_fail and p.tau > tol_fail)  std::cout << "UNKNOWN (tau > 0, kap > 0)";
    // else if(p.kap > tol_fail){
    //     bool pinfeas = model.h.dot(p.z) + model.b.dot(p.y) < RealScalar(0);
    //     bool dinfeas = model.c.dot(p.x) < RealScalar(0);
    //     if(pinfeas and dinfeas) std::cout << "PRIMAL_DUAL_INFEASIBLE";
    //     else if(pinfeas) std::cout << "PRIMAL_INFEASIBLE";
    //     else if(dinfeas) std::cout << "DUAL_INFEASIBLE";
    //     else    std::cout << "INFEASIBLE";
    // }else if(p.tau > tol_fail)  std::cout << "OPTIMAL";
    // else   std::cout << "UNEXPECTED";
    // std::cout << std::endl;
    return p;
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
    // q.x = - model.A.transpose() * p.y - model.G.transpose() * p.z - model.c;
    // q.y = model.A * p.x - model.b;
    // q.z = model.G * p.x - model.h + p.s;
    // q.tau = model.c.dot(p.x) + model.b.dot(p.y) + model.h.dot(p.z) + RealScalar(1);
    q.x = - model.c;
    q.x.noalias() -= model.G.transpose() * p.z;
    q.y = - model.b;
    q.z = - model.h + p.s;
    q.tau = model.h.dot(p.z) + RealScalar(1);
    calc_nu(model);
    alpha_sched = {1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.01, 0.001};
    // alpha_sched = {1.0, 0.99, 0.98, 0.97, 0.96};
}

// unused

template<typename RealScalar>
void Solver<RealScalar>::calc_nu(Model<RealScalar>& model){
    int barrier_parameter = 1 + model.cone().barrierParameter();
    nu = RealScalar(0.125) / (RealScalar(1) + std::sqrt(barrier_parameter));
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
Eigen::Vector<RealScalar, Eigen::Dynamic> Solver<RealScalar>::compute_newton_residuals(Model<RealScalar>& model) const{
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
    A.segment(0, model.n) = dp.x;
    A.segment(model.n, model.p) = dp.y;
    A.segment(model.n + model.p, model.d) = dp.z;
    A(model.n + model.p + model.d) = dp.tau;
    A(model.n + model.p + model.d + 1) = dp.theta;
    B.segment(model.n + model.p, model.d) = dp.s;
    B(model.n + model.p + model.d) = dp.kap;
    return T * A - B;
}

#endif