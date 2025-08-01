
// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE
#include "reader.cpp"
#include "solver.cpp"
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

using SOLVER_TYPE = mpfr::mpreal;

void mpfr_setup(){
    const int working_digits = 30;
    const int printing_digits = 6;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

int main(int argc, char* argv[]){
    if(argc < 2){
        throw std::logic_error("Too few arguments: Argument format is <input file>");
    }else if(argc > 2){
        throw std::logic_error("Too many arguments: Argument format is <input file>");
    }
    std::filesystem::path input_filepath(argv[1]);
    mpfr_setup();
    std::cout << "Reading model from " << input_filepath << std::endl;
    auto file_read_start = high_resolution_clock::now();
    Model<SOLVER_TYPE> model = reader<SOLVER_TYPE>(input_filepath);
    auto file_read_end = high_resolution_clock::now();
    model.print_model();
    std::cout << "Model read successfully! Now solving..." << std::endl;
    Solver<SOLVER_TYPE> solver;
    // Point<mpfr::mpreal> final_point = solver.solve(model, mpfr::mpreal("1e-6"), mpfr::mpreal("1e-8"));
    auto solve_start = high_resolution_clock::now();
    Point<SOLVER_TYPE> final_point = solver.solve(model, 1e-6, 1e-8);
    auto solve_end = high_resolution_clock::now();
    const int final_digits = 10;
    std::cout.precision(final_digits);
    std::cout << "tau = " << final_point.tau << std::endl;
    std::cout << "kap = " << final_point.kap << std::endl;
    std::cout << "Primal objective = " << model.c.dot(final_point.x) / final_point.tau << std::endl;
    std::cout << "Dual objective = " << - (model.h.dot(final_point.z) + model.b.dot(final_point.y)) / final_point.tau << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "File read time: " << duration_cast<milliseconds>(file_read_end - file_read_start).count() << "ms" << std::endl;
    std::cout << "Solve time: " << duration_cast<milliseconds>(solve_end - solve_start).count() << "ms" << std::endl;
}