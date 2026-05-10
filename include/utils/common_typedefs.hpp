#ifndef COMMON_TYPEDEFS_PAPP_VARGA_H
#define COMMON_TYPEDEFS_PAPP_VARGA_H
#include <Eigen/Core>

template<typename prec_type>
using optVector = Eigen::Vector<prec_type, Eigen::Dynamic>;
template<typename prec_type>
using optMatrix = Eigen::Matrix<prec_type, Eigen::Dynamic, Eigen::Dynamic>;

#endif