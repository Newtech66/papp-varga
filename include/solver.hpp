#ifndef SOLVER_PAPP_VARGA_H
#define SOLVER_PAPP_VARGA_H
#include <chrono>
#include <boost/math/tools/roots.hpp>
#include <Eigen/Core>
#include "linsolver.hpp"

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
    Point<prec_type> solve(Model<prec_type>& model, const prec_type& tol_gap,
        const prec_type& tol_fail, const int max_steps = 1000);
};

#endif