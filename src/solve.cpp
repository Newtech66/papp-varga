#include "esde_state.hpp"
#include "problem_data.hpp"

template<typename prec_type, class SteppingStrategy, class InitPoint, class TerminationCriteria>
ESDEState<prec_type> solve(const ProblemData<prec_type>& problem_data){
    ESDEState<prec_type> esde_state;
    InitPoint::set_init_point(esde_state);
    while(TerminationCriteria::terminate(esde_state, problem_data) == "not converged"){
        esde_state += SteppingStrategy.step(esde_state, problem_data);
    }
    return esde_state;
}