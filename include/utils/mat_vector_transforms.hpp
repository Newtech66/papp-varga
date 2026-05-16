#ifndef VECTORIZE_UTILS_H
#define VECTORIZE_UTILS_H
#include <complex>
#include <type_traits>
#include <cmath>
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename prec_type>
optVector<prec_type> split(const Eigen::Ref<const optVector<std::complex<prec_type>>>& vec){
    optVector<prec_type> out(2 * vec.size());
    out << vec.real(), vec.imag();
    return out;
}
template<typename prec_type>
optVector<std::complex<prec_type>> unsplit(const Eigen::Ref<const optVector<prec_type>>& vec){
    optVector<std::complex<prec_type>> out(vec.rows() / 2);
    out.real() = vec(Eigen::seqN(0, vec.rows() / 2));
    out.imag() = vec(Eigen::placeholders::lastN(vec.rows() / 2));
    return out;
}
template<typename prec_type>
optMatrix<prec_type> vec_to_sym_mat(const Eigen::Ref<const optVector<prec_type>>& vec, int mat_size){
    optMatrix<prec_type> out(mat_size, mat_size);
    for(int i = 0; i < mat_size; i++){
        out(i, i) = vec(i);
    }
    for(int i = 0, idx = mat_size; i < mat_size; i++){
        for(int j = 0; j < i; j++, idx++){
            out(i, j) = out(j, i) = vec(idx) / std::sqrt(2.0);
        }
    }
    return out;
}
template<typename prec_type>
optMatrix<std::complex<prec_type>> vec_to_her_mat(const Eigen::Ref<const optVector<prec_type>>& vec, int mat_size){
    optVector<std::complex<prec_type>> cvec = unsplit<prec_type>(vec);
    optMatrix<std::complex<prec_type>> out(mat_size, mat_size);
    for(int i = 0; i < mat_size; i++){
        out(i, i) = cvec(i);
    }
    for(int i = 0, idx = mat_size; i < mat_size; i++){
        for(int j = 0; j < i; j++, idx++){
            out(i, j) = out(j, i) = cvec(idx) / std::sqrt(2.0);
        }
    }
    return out;
}
template<typename prec_type>
optVector<prec_type> sym_mat_to_vec(const Eigen::Ref<const optMatrix<prec_type>>& mat, int mat_size){
    optVector<prec_type> out(mat_size * (mat_size + 1) / 2);
    for(int i = 0; i < mat_size; i++){
        out(i) = mat(i, i);
    }
    for(int i = 0, idx = mat_size; i < mat_size; i++){
        for(int j = 0; j < i; j++, idx++){
            out(idx) = std::sqrt(2.0) * mat(i, j);
        }
    }
    return out;
}
template<typename prec_type>
optVector<prec_type> her_mat_to_vec(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& mat, int mat_size){
    optVector<std::complex<prec_type>> out(mat_size * (mat_size + 1) / 2);
    for(int i = 0; i < mat_size; i++){
        out(i) = mat(i, i);
    }
    for(int i = 0, idx = mat_size; i < mat_size; i++){
        for(int j = 0; j < i; j++, idx++){
            out(idx) = std::sqrt(2.0) * mat(i, j);
        }
    }
    return split<prec_type>(out);
}
template<typename prec_type>
optVector<prec_type> her_mat_to_vec(const Eigen::Ref<const optMatrix<prec_type>>& mat, int mat_size){
    optVector<std::complex<prec_type>> out(mat_size * (mat_size + 1) / 2);
    for(int i = 0; i < mat_size; i++){
        out(i) = mat(i, i);
    }
    for(int i = 0, idx = mat_size; i < mat_size; i++){
        for(int j = 0; j < i; j++, idx++){
            out(idx) = std::sqrt(2.0) * mat(i, j);
        }
    }
    return split<prec_type>(out);
}

#endif