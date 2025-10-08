#ifndef SOLVER_PAPP_VARGA_H
#define SOLVER_PAPP_VARGA_H
#include <chrono>
#include <boost/math/tools/roots.hpp>
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
    const std::uintmax_t max_iter = 10;
    boost::math::tools::eps_tolerance<RealScalar> tcond = boost::math::tools::eps_tolerance<RealScalar>();
    void set_init_point(Model<RealScalar>& model);
    void print_header() const;
    void print_row(Model<RealScalar>& model, int, const std::chrono::duration<double>&, const std::chrono::duration<double>&) const;
    void calc_nu(Model<RealScalar>& model);
    RealScalar calc_iterate_norm(Model<RealScalar>& model, const RealScalar& mu);
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
    std::cout << std::left << std::setw(14) << "upd time (s)";
    std::cout << std::left << std::setw(12) << "mat time (s)";
    std::cout << std::endl;
}
template<typename RealScalar>
void Solver<RealScalar>::print_row(Model<RealScalar>& model, int istep, const std::chrono::duration<double>& upd_time, const std::chrono::duration<double>& mat_time) const{
    std::cout << std::left << std::setw(8)  << steps_taken;
    std::cout << std::left << std::setw(16) << std::scientific << std::setprecision(6) << model.c.dot(p.x) / p.tau;
    std::cout << std::left << std::setw(16) << std::scientific << std::setprecision(6) << (- model.h.dot(p.z) - model.b.dot(p.y)) / p.tau;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << p.tau;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << p.kap;
    std::cout << std::left << std::setw(14) << std::scientific << std::setprecision(4) << mu;
    std::cout << std::left << std::setw(8)  << istep;
    std::cout << std::left << std::setw(14) << std::fixed << std::setprecision(3) << upd_time.count();
    std::cout << std::left << std::setw(12) << std::fixed << std::setprecision(3) << mat_time.count();
    std::cout << std::endl;
}

template<typename RealScalar>
Point<RealScalar> Solver<RealScalar>::solve(Model<RealScalar>& model, const RealScalar& tol_gap, const RealScalar& tol_fail, const int max_steps){
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
    set_init_point(model);
    mu = RealScalar(1);
    LinearSolver solver(model, q);
    auto neighbourhood_check = [&](RealScalar c){
        dp = solver.solve_ns(model, p, q, c * mu, false);
        p += dp;
        model.cone().updatePoint(p.s);
        RealScalar diff = c * mu / RealScalar(4) - calc_iterate_norm(model, c * mu);
        p -= dp;
        model.cone().updatePoint(p.s);
        return diff;
    };
    steps_taken = 0;
    print_header();
    std::chrono::duration<double> total_time(0);
    while(p.s.dot(p.z) + p.tau * p.kap > tol_gap){
        if(steps_taken >= max_steps){
            std::cout << "Exiting because max iterations were reached" << std::endl;
            break;
        }
        //largest update
        std::uintmax_t istep = max_iter;
        auto mat_start = std::chrono::high_resolution_clock::now();
        solver.compute_aux_matrices(model, q);
        auto mat_end = std::chrono::high_resolution_clock::now();
        auto upd_start = std::chrono::high_resolution_clock::now();
        auto [loc, hic] = boost::math::tools::bracket_and_solve_root(neighbourhood_check, RealScalar(1) - nu, RealScalar(2.0), true, tcond, istep);
        mu *= hic;
        dp = solver.solve_ns(model, p, q, mu, false);
        p += dp;
        model.cone().updatePoint(p.s);
        // mu *= RealScalar(1) - nu;
        auto upd_end = std::chrono::high_resolution_clock::now();
        ++steps_taken;
        total_time += upd_end - upd_start + mat_end - mat_start;
        print_row(model, istep, upd_end - upd_start, mat_end - mat_start);
    }
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Iterations taken = " << steps_taken << std::endl;
    std::cout << "Solve time = " << std::fixed << std::setprecision(3) << total_time.count() << "s" << std::endl;
    return p;
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
}

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

#endif