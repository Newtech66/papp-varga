#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/MPRealSupport>
#include "model.cpp"
#include "solver.cpp"
#include "cones.cpp"
#include "possemidefinite.cpp"

void mpfr_setup(){
    const int working_digits = 60;
    const int printing_digits = 10;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

int main(){
    mpfr_setup();
    using SOLVER_TYPE = mpfr::mpreal;
    using MatrixXmp = Eigen::Matrix<SOLVER_TYPE, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXmp = Eigen::Vector<SOLVER_TYPE, Eigen::Dynamic>;
    int n = 3, p = 0, d = 4;
    // of the following, c is not allowed to be complex, the rest may be
    VectorXmp c(n), b(p), h(d);
    MatrixXmp A(p, n), G(d, n);
    c << 48, -8, 20;
    h << 11, 0, 0, -23;
    G << -10, 0, 0,
         -4, 0, 8,
         -4, 0, 8,
         0, 8, 2;
    std::vector<std::unique_ptr<Cone<SOLVER_TYPE>>> cones;
    PositiveSemidefinite<SOLVER_TYPE, false> cone1(2);
    cones.emplace_back(std::make_unique<decltype(cone1)>(cone1));
    Model<SOLVER_TYPE> model(c, A, G, b, h, cones);
    Solver<SOLVER_TYPE> solver;
    Point<SOLVER_TYPE> final_point = solver.solve(model, SOLVER_TYPE("0.00000001"), SOLVER_TYPE("0.0000001"));
    std::cout << "tau = " << final_point.tau << std::endl;
    std::cout << "kap = " << final_point.kap << std::endl;
    std::cout << "Primal objective = " << c.dot(final_point.x) / final_point.tau << std::endl;
    std::cout << "Dual objective = " << - (h.dot(final_point.z) + b.dot(final_point.y)) / final_point.tau << std::endl;
    model.print_model();
}