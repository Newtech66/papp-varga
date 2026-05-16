#ifndef ESDE_NEWTON_SOLVER_ESDE_H
#define ESDE_NEWTON_SOLVER_ESDE_H
#include <Eigen/Cholesky>
#include "common_typedefs.hpp"
#include "problem_data.hpp"
#include "esde_state.hpp"

/// @brief Class that solves the ESDE Newton system.
/// @tparam prec_type Underlying precision type of the solver.
/// @tparam SteppingStrategy The stepping strategy (Nesterov-Todd, Skajaa-Ye, etc.).
template<typename prec_type, typename SteppingStrategy>
// use of CRTP to get coupling between ESDENewtonSolver and stepping strategy
class ESDENewtonSolver{
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::LDLT<optMatrix<prec_type>> lltA, lltG;
    // The hessian matrix times G
    optMatrix<prec_type> HG;
public:
    /// @brief Solves the ESDE Newton system given a right hand side.
    /// @param esde_state The current state.
    /// @param problem_data The problem data.
    /// @param rhs The right hand side to solve for.
    /// @return The solution of the ESDE Newton system for the given rhs.
    ESDEState<prec_type> solve_newton_system(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& rhs, bool recompute){
        // it is assumed that update_auxiliary_matrices has been called at some point before this
        // otherwise answers will be wrong
        // compute first and second RHSes
        if(recompute = true)    update_auxiliary_matrices(problem_data);
        ESDEState<prec_type> sub_rhs1{rhs.x, rhs.y, static_cast<SteppingStrategy&>(*this).hvp(rhs.z) + rhs.s};
        ESDEState<prec_type> b1 = solve_subsystem(problem_data, sub_rhs1);
        ESDEState<prec_type> sub_rhs2{problem_data.c, problem_data.b, static_cast<SteppingStrategy&>(*this).hvp(problem_data.h, problem_data)};
        ESDEState<prec_type> b2 = solve_subsystem(problem_data, sub_rhs2);
        ESDEState<prec_type> out;
        out.tau = (rhs.tau + rhs.kap / esde_state.tau +
            problem_data.c.dot(sub_rhs1.x) + problem_data.b.dot(sub_rhs1.y) +
            problem_data.h.dot(sub_rhs1.z)) / (esde_state.kap / esde_state.tau +
            problem_data.c.dot(sub_rhs2.x) + problem_data.b.dot(sub_rhs2.y) +
            problem_data.h.dot(sub_rhs2.z));
        out.x = sub_rhs1.x - out.tau * sub_rhs2.x;
        out.y = sub_rhs1.y - out.tau * sub_rhs2.y;
        out.z = sub_rhs1.z - out.tau * sub_rhs2.z;
        out.s = -problem_data.G * out.x + out.tau * problem_data.h - rhs.z;
        out.kap = (rhs.kap - esde_state.kap * out.tau) / esde_state.tau;
        return out;
    }
    /// @brief Updates the internal structures of the solver for re-use across calls to solve().
    /// @param problem_data The problem data.
    void update_auxiliary_matrices(ProblemData<prec_type>& problem_data){
        // only resized on the first call and is a no-op on all subsequent calls
        HG.resize(problem_data.G.rows(), problem_data.G.cols());
        HG = static_cast<SteppingStrategy&>(*this).hvp(problem_data.G, problem_data);
        lltG.compute(problem_data.G.transpose() * HG);
        lltA.compute(problem_data.A * lltG.solve(problem_data.A.transpose()));
    }
    /// @brief Solves the 3x3 block subsystem given a right hand side.
    /// @param problem_data The problem data.
    /// @param rhs The right hand side to solve for.
    /// @return The solution of the 3x3 block subsystem for the given rhs.
    ESDEState<prec_type> solve_subsystem(ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& rhs){
        ESDEState<prec_type> out;
        optVector<prec_type> xGz = rhs.x - problem_data.G.transpose() * rhs.z;
        out.y = lltA.solve(rhs.y + problem_data.A * lltG.solve(xGz));
        out.x = lltG.solve(xGz - problem_data.A.transpose() * out.y);
        out.z = rhs.z + HG * out.x;
        return out;
    }
};

#endif