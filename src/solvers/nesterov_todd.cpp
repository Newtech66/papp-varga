#include "solvers/nesterov_todd.hpp"

template<typename prec_type>
optVector<prec_type> NesterovToddSolver<prec_type>::findPredictionDirection(){

}
template<typename prec_type>
prec_type NesterovToddSolver<prec_type>::findCenteringParameter(){

}
template<typename prec_type>
prec_type NesterovToddSolver<prec_type>::findLargestStepSize(){

}
template<typename prec_type>
optVector<prec_type> NesterovToddSolver<prec_type>::findCombinedDirection(){

}

template<typename prec_type>
void NesterovToddSolver<prec_type>::updateInternalState(){
    int cpos = 0;
    for(int i = 0; i < esde_state.problem_data->cones.length(); i++){
        auto& cone = esde_state->problem_data->cones[i];
        int npar = cone.numParams();
        auto idxs = Eigen::seqN(start, size);
        if(cone.coneId() == "PSD"){
            auto Sk = vec_to_her_mat(esde_state.s(idxs));
            auto Zk = vec_to_her_mat(esde_state.z(idxs));
            if(is_complex){
                // Find scaling point
                her_eigsolver.compute(Sk);
                auto Skhalf = her_eigsolver.operatorSqrt();
                auto Skihalf = her_eigsolver.operatorInverseSqrt();
                her_eigsolver.compute(Skhalf * Zk * Skhalf);
                scaling_point(idxs) = her_mat_to_vec(Skihalf * her_eigsolver.operatorSqrt() * Skihalf);
                // Find scaling matrix and lambda
                auto L1 = Sk.llt().matrixL();
                auto L2 = Zk.llt().matrixL();
                her_svd.compute(L2.conj() * L1);
                scaling_matrix(idxs) = her_mat_to_vec(L * her_svd.matrixV() * her_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal());
                lambda(idxs) = her_svd.singularValues();
            }else{
                // Find scaling point
                sym_eigsolver.compute(Sk);
                auto Skhalf = sym_eigsolver.operatorSqrt();
                auto Skihalf = sym_eigsolver.operatorInverseSqrt();
                sym_eigsolver.compute(Skhalf * Zk * Skhalf);
                scaling_point(idxs) = sym_mat_to_vec(Skihalf * sym_eigsolver.operatorSqrt() * Skihalf);
                // Find scaling matrix and lambda
                auto L1 = Sk.llt().matrixL();
                auto L2 = Zk.llt().matrixL();
                sym_svd.compute(L2.conj() * L1);
                scaling_matrix(idxs) = sym_mat_to_vec(L * sym_svd.matrixV() * sym_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal());
                lambda(idxs) = sym_svd.singularValues();
            }
        }else{
            throw std::logic_error("Received invalid cone");
        }
        cpos += npar;
    }
}
template<typename prec_type>
void NesterovToddSolver<prec_type>::step(){

}
