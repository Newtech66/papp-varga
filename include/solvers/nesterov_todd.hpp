#ifndef NESTEROV_TODD_SOLVERS_H
#define NESTEROV_TODD_SOLVERS_H
#include "common_typedefs.hpp"
#include "esde_state.hpp"
#include "problem_data.hpp"
#include "esde_newton_solver.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>

template<typename prec_type>
class NesterovTodd{
private:
    optVector<prec_type> scaling_matrix; // this is the scaling matrix
    optVector<prec_type> lambda; // this is lambda = W * z = (W^-1)^T * s
    optVector<prec_type> scaling_point; // this is the scaling point

    Eigen::SelfAdjointEigenSolver<prec_type> sym_eigsolver;
    Eigen::SelfAdjointEigenSolver<std::complex<prec_type>> her_eigsolver;

    Eigen::BDCSVD<prec_type, Eigen::ComputeThinV> sym_svd;
    Eigen::BDCSVD<std::complex<prec_type>, Eigen::ComputeThinV> her_svd;

    ESDENewtonSolver<prec_type> esde_newton_solver;
    prec_type mu;

    optVector<prec_type> findPredictionDirection(const ESDEState<prec_type>&, const ProblemData<prec_type>&);
    optVector<prec_type> findCombinedDirection(const ESDEState<prec_type>&, const ProblemData<prec_type>&, const ESDEState<prec_type>&, prec_type);
    prec_type findLargestStepSize(const ESDEState<prec_type>&, const ProblemData<prec_type>&, const ESDEState<prec_type>&);

public:
    NesterovTodd() = default;
    void updateInternalState(const ESDEState<prec_type>&, const ProblemData<prec_type>&);
    optVector<prec_type> step(const ESDEState<prec_type>&, const ProblemData<prec_type>&);
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&);
};

template<typename prec_type>
prec_type NesterovTodd<prec_type>::findLargestStepSize(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& dir){
    std::vector<prec_type> alphak;
    int cpos = 0;
    for(auto& [cone_id, cone_params]: problem_data.cones){
        int nvar = cone_params.numVariables();
        auto idxs = Eigen::seqN(cpos, nvar);
        if(cone_id == "REALPSD"){
            auto lis = lambda(idxs).cwiseInverse().cwiseSqrt().asDiagonal();
            auto rhok = lis * vec_to_sym_mat(dir.s(idxs)) * lis;
            auto sigk = lis * vec_to_sym_mat(dir.z(idxs)) * lis;
            sym_eigsolver.compute(rhok);
            prec_type gams = sym_eigsolver.eigenvalues()(0);
            sym_eigsolver.compute(sigk);
            prec_type gamz = sym_eigsolver.eigenvalues()(0);
            alphak.push_back(1 / std::max({0, -gams, -gamz}));
        }else if(cone_id == "COMPLEXPSD"){
            auto lis = lambda(idxs).cwiseInverse().cwiseSqrt().asDiagonal();
            auto rhok = lis * vec_to_her_mat(dir.s(idxs)) * lis;
            auto sigk = lis * vec_to_her_mat(dir.z(idxs)) * lis;
            her_eigsolver.compute(rhok);
            prec_type gams = her_eigsolver.eigenvalues()(0);
            her_eigsolver.compute(sigk);
            prec_type gamz = her_eigsolver.eigenvalues()(0);
            alphak.push_back(1 / std::max({0, -gams, -gamz}));
        }else{
            throw std::invalid_argument("Received invalid cone in method findLargestStepSize of NesterovTodd stepper");
        }
        cpos += nvar;
    }
    return *std::min_element(alphak.begin(), alphak.end());
}
template<typename prec_type>
optVector<prec_type> NesterovTodd<prec_type>::findPredictionDirection(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data){
    ESDEState<prec_type> dir;
    dir.x = problem_data.A.transpose() * esde_state.y + problem_data.G.transpose() * esde_state.z + problem_data.c * esde_state.tau;
    dir.y = -problem_data.A * esde_state.x + problem_data.b * esde_state.tau;
    dir.z = -problem_data.G * esde_state.x + problem_data.h * esde_state.tau - esde_state.s;
    dir.tau = -problem_data.c.dot(esde_state.x) - problem_data.b.dot(esde_state.y) - problem_data.h.dot(esde_state.z) - esde_state.kap;
    dir.x *= prec_type(-1);
    dir.y *= prec_type(-1);
    dir.z *= prec_type(-1);
    dir.tau *= prec_type(-1);
    dir.s = -esde_state.z;
    dir.kap = -esde_state.tau * esde_state.kap;
    return esde_newton_solver.solve(problem_data, dir, *this);
}
template<typename prec_type>
optVector<prec_type> NesterovTodd<prec_type>::findCombinedDirection(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data, const ESDEState<prec_type>& pred_dir, prec_type centering_parameter){
    dir.x = problem_data.A.transpose() * esde_state.y + problem_data.G.transpose() * esde_state.z + problem_data.c * esde_state.tau;
    dir.y = -problem_data.A * esde_state.x + problem_data.b * esde_state.tau;
    dir.z = -problem_data.G * esde_state.x + problem_data.h * esde_state.tau - esde_state.s;
    dir.tau = -problem_data.c.dot(esde_state.x) - problem_data.b.dot(esde_state.y) - problem_data.h.dot(esde_state.z) - esde_state.kap;
    dir.x *= prec_type(1) - centering_parameter;
    dir.y *= prec_type(1) - centering_parameter;
    dir.z *= prec_type(1) - centering_parameter;
    dir.tau *= prec_type(1) - centering_parameter;
    dir.s.resizeLike(esde_state.s);
    int cpos = 0;
    for(auto& [cone_id, cone_params]: problem_data.cones){
        int nvar = cone_params.numVariables();
        auto idxs = Eigen::seqN(cpos, nvar);
        if(cone_id == "REALPSD"){
            auto W = vec_to_sym_mat(scaling_matrix(idxs));
            sym_eigsolver.compute(W);
            dir.s(idxs) = W.transpose() * (-lambda(idxs) - PositiveSemidefinite<prec_type, false>::diamond_product(lambda(idxs), sym_eigsolver.operatorInvers))
        }else if(cone_id == "COMPLEXPSD"){

        }else{
            throw std::invalid_argument("Received invalid cone in method findCombinedDirection of NesterovTodd stepper");
        }
        cpos += nvar;
    }
    dir.kap = -esde_state.tau * esde_state.kap -pred_dir.tau * pred_dir.kap + centering_parameter * mu;
    return esde_newton_solver.solve(problem_data, dir, *this);
}
template<typename prec_type>
optVector<prec_type> NesterovTodd<prec_type>::step(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data){
    updateInternalState(esde_state, problem_data);
    optVector<prec_type> pred_dir = findPredictionDirection(esde_state, problem_data);
    prec_type alpha_p = findLargestStepSize(esde_state, problem_data, pred_dir);
    prec_type centering_parameter = (prec_type(1) - alpha_p) * (prec_type(1) - alpha_p) * (prec_type(1) - alpha_p);
    optVector<prec_type> comb_dir = findCombinedDirection(esde_state, problem_data, centering_parameter);
    prec_type alpha = findLargestStepSize(esde_state, problem_data, comb_dir);
    return prec_type(0.99) * alpha * comb_dir;
}
template<typename prec_type>
void NesterovTodd<prec_type>::updateInternalState(const ESDEState<prec_type>& esde_state, const ProblemData<prec_type>& problem_data){
    int cpos = 0, m = 0;
    for(auto& [cone_id, cone_params]: problem_data.cones){
        int nvar = cone_params.numVariables();
        auto idxs = Eigen::seqN(cpos, nvar);
        if(cone_id == "REALPSD"){
            auto Sk = vec_to_sym_mat(esde_state.s(idxs));
            auto Zk = vec_to_sym_mat(esde_state.z(idxs));
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
            scaling_matrix(idxs) = sym_mat_to_vec(L1 * sym_svd.matrixV() * sym_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal());
            lambda(idxs) = sym_svd.singularValues();
        }else if(cone_id == "COMPLEXPSD"){
            auto Sk = vec_to_her_mat(esde_state.s(idxs));
            auto Zk = vec_to_her_mat(esde_state.z(idxs));
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
            scaling_matrix(idxs) = her_mat_to_vec(L1 * her_svd.matrixV() * her_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal());
            lambda(idxs) = her_svd.singularValues();
        }else{
            throw std::invalid_argument("Received invalid cone in method updateInternalState of NesterovTodd stepper");
        }
        cpos += nvar;
        m += cone_params.barrierParameter();
    }
    mu = (lambda.dot(lambda) + esde_state.tau * esde_state.kap) / (m + 1);
}

#endif