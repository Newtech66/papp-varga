#include "reader.hpp"
#include "possemidefinite.hpp"
#include "logperspecepi.hpp"
#include "common_typedefs.hpp"
#include <fstream>

template<typename prec_type>
std::unique_ptr<Cone<prec_type>> get_cone(const std::string& cone_name, const int cone_size){
    if(cone_name == "REALPSD"){
        return std::make_unique<PositiveSemidefinite<prec_type, false>>(cone_size);
    }else if(cone_name == "COMPLEXPSD"){
        return std::make_unique<PositiveSemidefinite<prec_type, true>>(cone_size);
    }else if(cone_name == "DIAGONALPSD"){
        return std::make_unique<DiagonalPositiveSemidefinite<prec_type>>(cone_size);
    }else if(cone_name == "REALLOGPERSPECEPI"){
        return std::make_unique<LogPerspecEpi<prec_type, false>>(cone_size);
    }else if(cone_name == "COMPLEXLOGPERSPECEPI"){
        return std::make_unique<LogPerspecEpi<prec_type, true>>(cone_size);
    }
    const std::string error_message = cone_name + " is an unsupported cone type!";
    throw std::logic_error(error_message);
}

template<typename prec_type>
Model<prec_type> reader(const std::filesystem::path& input_filepath){
    // Open the file
    std::ifstream input_file(input_filepath);
    if(!input_file){
        throw std::logic_error("File could not be opened!");
    }
    int n, p, k, d = 0;
    // Read n, p, k
    input_file >> n >> p >> k;
    // Read cones
    cone_array<prec_type> cones;
    for(int i = 0; i < k; ++i){
        std::string cone_name;
        input_file >> cone_name;
        int cone_size;
        input_file >> cone_size;
        cones.emplace_back(get_cone<prec_type>(cone_name, cone_size));
        d += cones.back()->numParams();
    }
    // Read c
    optMatrix<prec_type> A(p, n), G(d, n);
    optVector<prec_type> c(n), b(p), h(d);
    for(int i = 0; i < n; ++i){
        input_file >> c(i);
    }
    for(int i = 0; i < p; ++i){
        for(int j = 0; j < n; ++j){
            input_file >> A(i, j);
        }
    }
    for(int i = 0; i < p; ++i){
        input_file >> b(i);
    }
    for(int i = 0; i < d; ++i){
        for(int j = 0; j < n; ++j){
            input_file >> G(i, j);
        }
    }
    for(int i = 0; i < d; ++i){
        input_file >> h(i);
    }
    // attempt error check
    if(input_file.eof()){
        throw std::logic_error("Unexpected end of file!");
    }
    std::string e;
    if(input_file >> e){
        throw std::logic_error("More data in file than expected!");
    }
    return Model<prec_type>(c, A, b, G, h, cones);
}

template std::unique_ptr<Cone<double>> get_cone(const std::string&, const int);
template Model<double> reader(const std::filesystem::path&);
