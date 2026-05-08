#ifndef READER_PAPP_VARGA_H
#define READER_PAPP_VARGA_H
#include <filesystem>
#include "model.hpp"

template<typename prec_type>
std::unique_ptr<Cone<prec_type>> get_cone(const std::string& cone_name, const int cone_size);
template<typename prec_type>
Model<prec_type> reader(const std::filesystem::path& input_filepath);

#endif