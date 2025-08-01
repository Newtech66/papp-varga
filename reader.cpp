#ifndef READER_PAPP_VARGA_H
#define READER_PAPP_VARGA_H
#include <fstream>
#include <filesystem>
#include <map>
#include <unsupported/Eigen/MPRealSupport>
#include "model.cpp"
#include "possemidefinite.cpp"
#include "logperspecepi.cpp"

template<typename T>
std::unique_ptr<Cone<T>> get_cone(const std::string& cone_name, const int cone_size){
    if(cone_name == "REALPSD"){
        using cone = PositiveSemidefinite<T, false>;
        return std::make_unique<cone>(cone(cone_size));
    }else if(cone_name == "COMPLEXPSD"){
        using cone = PositiveSemidefinite<T, true>;
        return std::make_unique<cone>(cone(cone_size));
    }else if(cone_name == "DIAGONALPSD"){
        using cone = DiagonalPositiveSemidefinite<T>;
        return std::make_unique<cone>(cone(cone_size));
    }
    // else if(cone_name == "REALLOGPERSPECEPI"){
    //     using cone = LogPerspecEpi<T, false>;
    //     return std::make_unique<cone>(cone(cone_size));
    // }else if(cone_name == "COMPLEXLOGPERSPECEPI"){
    //     using cone = LogPerspecEpi<T, true>;
    //     return std::make_unique<cone>(cone(cone_size));
    // }
    const std::string error_message = cone_name + " is an unsupported cone type!";
    throw std::logic_error(error_message);
}

template<typename T>
Model<T> reader(const std::filesystem::path& input_filepath){
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<T, Eigen::Dynamic>;
    // Open the file
    std::ifstream input_file(input_filepath);
    if(!input_file){
        throw std::logic_error("File could not be opened!");
    }
    int n, p, k, d = 0;
    // Read n, p, k
    input_file >> n >> p >> k;
    // Read cones
    std::vector<std::unique_ptr<Cone<T>>> cones;
    for(int i = 0; i < k; ++i){
        std::string cone_name;
        input_file >> cone_name;
        int cone_size;
        input_file >> cone_size;
        cones.emplace_back(get_cone<T>(cone_name, cone_size));
        d += cones.back()->numParams();
    }
    // Read c
    Matrix A(p, n), G(d, n);
    Vector c(n), b(p), h(d);
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
    while(input_file >> e){
        throw std::logic_error("More data in file than expected!");
    }
    return Model<T>(c, A, b, G, h, cones);
}

#endif