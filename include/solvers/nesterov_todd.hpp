#ifndef NESTEROV_TODD_SOLVERS_H
#define NESTEROV_TODD_SOLVERS_H
#include <Eigen/Core>
#include "common_typedefs.hpp"
#include "esde_state.hpp"
#include "problem_data.hpp"
#include "esde_newton_solver.hpp"
#include <algorithm>

template<typename prec_type>
class NesterovTodd : public ESDENewtonSolver<prec_type, NesterovTodd<prec_type>>{
private:
    // Parameter for central path, starts at 1 and goes to 0 at optimum.
    prec_type mu;

    ESDEState<prec_type> findPredictionDirection(const ESDEState<prec_type>& esde_state, ProblemData<prec_type>& problem_data){
        ESDEState<prec_type> rhs;
        rhs.x = problem_data.A.transpose() * esde_state.y + problem_data.G.transpose() * esde_state.z + problem_data.c * esde_state.tau;
        rhs.y = -problem_data.A * esde_state.x + problem_data.b * esde_state.tau;
        rhs.z = -problem_data.G * esde_state.x + problem_data.h * esde_state.tau - esde_state.s;
        rhs.tau = -problem_data.c.dot(esde_state.x) - problem_data.b.dot(esde_state.y) - problem_data.h.dot(esde_state.z) - esde_state.kap;
        rhs.x *= prec_type(-1);
        rhs.y *= prec_type(-1);
        rhs.z *= prec_type(-1);
        rhs.tau *= prec_type(-1);
        rhs.s = -esde_state.z;
        rhs.kap = -esde_state.tau * esde_state.kap;
        return this->solve_newton_system(esde_state, problem_data, rhs);
    }
    ESDEState<prec_type> findCombinedDirection(const ESDEState<prec_type>& esde_state, ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& pred_dir, const prec_type& centering_parameter){
        ESDEState<prec_type> rhs;
        rhs.x = problem_data.A.transpose() * esde_state.y + problem_data.G.transpose() * esde_state.z + problem_data.c * esde_state.tau;
        rhs.y = -problem_data.A * esde_state.x + problem_data.b * esde_state.tau;
        rhs.z = -problem_data.G * esde_state.x + problem_data.h * esde_state.tau - esde_state.s;
        rhs.tau = -problem_data.c.dot(esde_state.x) - problem_data.b.dot(esde_state.y) - problem_data.h.dot(esde_state.z) - esde_state.kap;
        rhs.x *= prec_type(1) - centering_parameter;
        rhs.y *= prec_type(1) - centering_parameter;
        rhs.z *= prec_type(1) - centering_parameter;
        rhs.tau *= prec_type(1) - centering_parameter;
        rhs.s = problem_data.cones.get_nt_rhs_s(pred_dir.s, pred_dir.z, centering_parameter, mu);
        rhs.kap = -esde_state.tau * esde_state.kap - pred_dir.tau * pred_dir.kap + centering_parameter * mu;
        return this->solve_newton_system(esde_state, problem_data, rhs);
    }
public:
    NesterovTodd() = default;
    ESDEState<prec_type> step(const ESDEState<prec_type>& esde_state, ProblemData<prec_type>& problem_data){
        this->update_auxiliary_matrices(problem_data);
        problem_data.cones.update_nt_scaling(esde_state.s, esde_state.z);
        mu = (scaled_variable.dot(scaled_variable) + esde_state.tau * esde_state.kap) / (problem_data.cones.barrierParameter() + 1);
        ESDEState<prec_type> pred_dir = findPredictionDirection(esde_state, problem_data);
        prec_type alpha_p = problem_data.cones.get_nt_step_length(pred_dir.s, pred_dir.z, scaled_variable);
        prec_type centering_parameter = (prec_type(1) - alpha_p) * (prec_type(1) - alpha_p) * (prec_type(1) - alpha_p);
        ESDEState<prec_type> comb_dir = findCombinedDirection(esde_state, problem_data, pred_dir, centering_parameter);
        prec_type alpha = problem_data.cones.get_nt_step_length(comb_dir.s, comb_dir.z);
        return prec_type(0.99) * alpha * comb_dir;
    }
    optMatrix<prec_type> hvp(const Eigen::Ref<const optMatrix<prec_type>>& vecs, ProblemData<prec_type>& problem_data){
        return problem_data.cones.hvp(scaling_point, vecs);
    }
};

#endif