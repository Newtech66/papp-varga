#include <Eigen/Core>
#include "problem_data.hpp"
#include "esde_state.hpp"
#include "mat_vector_transforms.hpp"
#include "nesterov_todd.hpp"
#include "psd.hpp"
#include <iostream>
#include <format>

using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;

int main(){
    Vec c(6);
    c << 0, 0, 0, 1, 1, 0;
    Mat A(3, 6);
    A.setZero();
    A(0, 0) = A(1, 1) = A(2, 2) = 1;
    Vec b(3);
    b << 1, 1, 1;
    Mat G(6, 6);
    G.setIdentity();
    G *= -1;
    Vec h(6);
    h.setZero();
    ProblemData<double> pd(c, A, b, G, h);
    pd.print_problem_data();
    pd.cones.addCone<ConeTypes::REALPSD>(PSD<double, false>(3));
    ESDEState<double> esde_state;
    esde_state.kap = esde_state.tau = 1;
    esde_state.x = Vec::Zero(6);
    esde_state.y = Vec::Zero(3);
    esde_state.s = sym_mat_to_vec<double>(Mat::Identity(3, 3), 3);
    esde_state.z = sym_mat_to_vec<double>(Mat::Identity(3, 3), 3);
    NesterovTodd<double> nt;
    for(int i = 0; i < 5; i++){
        esde_state += nt.step(esde_state, pd);
        std::cout << std::format("Primal objective = {}\n", pd.c.dot(esde_state.x));
    }
}