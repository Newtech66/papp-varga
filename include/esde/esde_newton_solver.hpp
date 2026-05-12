#ifndef LINSOLVER_PAPP_VARGA_H
#define LINSOLVER_PAPP_VARGA_H
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include "common_typedefs.hpp"
#include "problem_data.hpp"
#include "esde_state.hpp"

/// @brief Class that solves the ESDE Newton system.
/// @tparam prec_type Underlying precision type of the solver.
template<typename prec_type>
class ESDENewtonSolver{
private:
    // it appears using LDLT is far more stable than LLT
    Eigen::LDLT<optMatrix<prec_type>> lltA, lltG;
public:
    /// @brief Updates the internal structures of the solver for re-use across calls to solve().
    /// @tparam SteppingStrategy The stepping strategy (Nesterov-Todd, Skajaa-Ye, etc.) that called the function.
    /// Required for calling out to the HVP provided by the stepping strategy.
    /// @param problem_data The problem data.
    /// @param stepper See SteppingStrategy.
    template<class SteppingStrategy>
    void update_auxiliary_matrices(const ProblemData<prec_type>& problem_data, SteppingStrategy& stepper);
    /// @brief Solves the ESDE Newton system given a right hand side.
    /// @tparam SteppingStrategy The stepping strategy (Nesterov-Todd, Skajaa-Ye, etc.) that called the function.
    /// Required for calling out to the HVP provided by the stepping strategy.
    /// @param problem_data The problem data.
    /// @param rhs The right hand side to solve for.
    /// @param stepper See SteppingStrategy.
    /// @return The solution of the ESDE Newton system for the given rhs.
    template<class SteppingStrategy>
    optVector<prec_type> solve(const ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& rhs, SteppingStrategy& stepper);
};

template<typename prec_type>
template<class SteppingStrategy>
optVector<prec_type> ESDENewtonSolver<prec_type>::solve(const ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& rhs, SteppingStrategy& stepper){
    
}

#endif