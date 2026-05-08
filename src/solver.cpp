#ifndef SOLVER_PAPP_VARGA_H
#define SOLVER_PAPP_VARGA_H
#include <chrono>
#include <boost/math/tools/roots.hpp>
#include <Eigen/Core>
// #include <mpreal.h>
#include "model.cpp"
#include "point.hpp"
#include "linsolver.cpp"

template<typename prec_type>
class Solver{
private:
    Point<prec_type> p, dp, q;
    prec_type mu, nu;
    int steps_taken;
    const std::uintmax_t max_iter = 10;
    boost::math::tools::eps_tolerance<prec_type> tcond = boost::math::tools::eps_tolerance<prec_type>();
    void set_init_point(Model<prec_type>& model);
    void print_header() const;
    void print_row(Model<prec_type>& model, int, const std::chrono::duration<double>&, const std::chrono::duration<double>&) const;
    void calc_nu(Model<prec_type>& model);
    prec_type calc_iterate_norm(Model<prec_type>& model, const prec_type& mu);
public:
    Point<prec_type> solve(Model<prec_type>& model, const prec_type& tol_gap, const prec_type& tol_fail, const int max_steps = 1000);
};

template<typename prec_type>
void Solver<prec_type>::print_header() const{
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
template<typename prec_type>
void Solver<prec_type>::print_row(Model<prec_type>& model, int istep, const std::chrono::duration<double>& upd_time, const std::chrono::duration<double>& mat_time) const{
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

template<typename prec_type>
Point<prec_type> Solver<prec_type>::solve(Model<prec_type>& model, const prec_type& tol_gap, const prec_type& tol_fail, const int max_steps){
    set_init_point(model);
    mu = prec_type(1);
    LinearSolver solver(model, q);
    auto neighbourhood_check = [&](prec_type c){
        dp = solver.solve_ns(model, p, q, c * mu, false);
        p += dp;
        model.cone().updatePoint(p.s);
        prec_type diff = c * mu / prec_type(4) - calc_iterate_norm(model, c * mu);
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
        // auto [loc, hic] = boost::math::tools::bracket_and_solve_root(neighbourhood_check, prec_type(1) - nu, prec_type(2.0), true, tcond, istep);
        // mu *= hic;
        dp = solver.solve_ns(model, p, q, mu, false);
        p += dp;
        model.cone().updatePoint(p.s);
        mu *= prec_type(1) - nu;
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

template<typename prec_type>
void Solver<prec_type>::set_init_point(Model<prec_type>& model){
    // x = 0, y = 0, z = -g(s), s = (set to an interior point of the cone)
    // kap = tau = theta = 1
    // x -> R[n, 1], y -> R[p, 1], z -> R[d, 1], s -> R[d, 1]
    p.x = optVector<prec_type>::Zero(model.n);
    p.y = optVector<prec_type>::Zero(model.p);
    p.s = model.cone().point();
    p.z = -model.cone().jacobian();
    p.kap = p.tau = p.theta = prec_type(1);
    // calculate q
    // q.x = - model.A.transpose() * p.y - model.G.transpose() * p.z - model.c;
    // q.y = model.A * p.x - model.b;
    // q.z = model.G * p.x - model.h + p.s;
    // q.tau = model.c.dot(p.x) + model.b.dot(p.y) + model.h.dot(p.z) + prec_type(1);
    q.x = - model.c;
    q.x.noalias() -= model.G.transpose() * p.z;
    q.y = - model.b;
    q.z = - model.h + p.s;
    q.tau = model.h.dot(p.z) + prec_type(1);
    calc_nu(model);
}

template<typename prec_type>
void Solver<prec_type>::calc_nu(Model<prec_type>& model){
    int barrier_parameter = 1 + model.cone().barrierParameter();
    nu = prec_type(0.125) / (prec_type(1) + std::sqrt(barrier_parameter));
}

template<typename prec_type>
prec_type Solver<prec_type>::calc_iterate_norm(Model<prec_type>& model, const prec_type& mu){
    // ||z + mu * g(s) + tau + mu * g(kap)||*_(s, kap)
    // kap and tau are in non-neg orthant (>=0)
    // the barrier function is -log(x), gradient is -1/x, hessian is 1/x^2
    optVector<prec_type> t = p.z + mu * model.cone().jacobian();
    // sqrt(t^ * H^-1 * t)
    prec_type n2 = t.dot(model.cone().ihvp(t));
    // add the kap tau part
    // (tau - mu / kap) * kap^2 (tau - mu / kap) = (kap * tau - mu) ** 2
    n2 += (p.kap * p.tau - mu) * (p.kap * p.tau - mu);
    return std::sqrt(n2);
}

// template<>
// mpfr::mpreal Solver<mpfr::mpreal>::calc_iterate_norm(Model<mpfr::mpreal>& model, const mpfr::mpreal& mu){
//     optVector<mpfr::mpreal> t = p.z + mu * model.cone().jacobian();
//     mpfr::mpreal n2 = t.dot(model.cone().ihvp(t));
//     n2 += (p.kap * p.tau - mu) * (p.kap * p.tau - mu);
//     return mpfr::sqrt(n2);
// }

#endif