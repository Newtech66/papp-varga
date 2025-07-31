#define EIGEN_NO_AUTOMATIC_RESIZING
#include "reader.cpp"
#include "solver.cpp"

void mpfr_setup(){
    const int working_digits = 60;
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
    Model<mpfr::mpreal> model = reader<mpfr::mpreal>(input_filepath);
    model.print_model();
    std::cout << "Model read successfully! Now solving..." << std::endl;
    Solver<mpfr::mpreal> solver;
    Point<mpfr::mpreal> final_point = solver.solve(model, mpfr::mpreal("1e-8"), mpfr::mpreal("1e-8"));
    std::cout << "tau = " << final_point.tau << std::endl;
    std::cout << "kap = " << final_point.kap << std::endl;
    std::cout << "Primal objective = " << model.c.dot(final_point.x) / final_point.tau << std::endl;
    std::cout << "Dual objective = " << - (model.h.dot(final_point.z) + model.b.dot(final_point.y)) / final_point.tau << std::endl;
}