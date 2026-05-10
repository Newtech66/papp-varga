#ifndef ESDE_ESDE_H
#define ESDE_ESDE_H

/// @brief Class representing the extended self-dual embedding.
///
/// This class holds the problem data, internal state, and the solver backend, which is all the
/// required information to solve a problem.
///
/// @tparam prec_type 
/// @tparam SolverBackend 
template<typename prec_type, class SolverBackend>
class ESDE{
private:
    ESDEState<prec_type> state; // stores the internal state of the ESDE + problem data
    SolverBackend<prec_type> solver; // the solver object
public:
    ESDE() = delete; // We cannot default initialize this class
    ESDE(const ProblemData<prec_type>& data){
        // TODO: There is a possible problem that I can't ensure that the state and solver
        // both have the same prec_type.
        state = ESDEState<prec_type>(data);
        solver = SolverBackend<prec_type>(state);
    }
    void solve(){
        
    }
};

#endif