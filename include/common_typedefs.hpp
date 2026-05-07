#ifndef COMMON_TYPEDEFS_PAPP_VARGA_H
#define COMMON_TYPEDEFS_PAPP_VARGA_H

template<typename RealScalar>
using optVector = Eigen::Vector<RealScalar, Eigen::Dynamic>;

template<typename RealScalar>
using optMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;

#endif