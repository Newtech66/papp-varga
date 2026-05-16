#ifndef PSD_CONES_H
#define PSD_CONES_H
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "psd_parameters.hpp"
#include "common_typedefs.hpp"
#include "mat_vector_transforms.hpp"

template<typename prec_type, bool is_complex>
class PSD{
public:
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    static optVector<prec_type> grad(const Eigen::Ref<const optVector<prec_type>>& p, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            return -her_mat_to_vec<prec_type>(vec_to_her_mat<prec_type>(p, mat_size).ldlt().solve(cone_params.iden), mat_size);
        }else{
            return -sym_mat_to_vec<prec_type>(vec_to_sym_mat<prec_type>(p, mat_size).ldlt().solve(cone_params.iden), mat_size);
        }
    }
    /// @brief Computes the hessian-vector product (HVP) F''(p)q.
    ///
    /// For the PSD cone, this is given by vec(mat(p)^-1 mat(q) mat(p)^-1).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the HVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    static optMatrix<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& q, const PSDParameters& cone_params){
        using namespace Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto pinv = vec_to_her_mat<prec_type>(p, mat_size).ldlt().solve(cone_params.iden);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(pinv * vec_to_her_mat<prec_type>(q(all, cidx), mat_size) * pinv, mat_size);
            }
        }else{
            auto pinv = vec_to_sym_mat<prec_type>(p, mat_size).ldlt().solve(cone_params.iden);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec<prec_type>(pinv * vec_to_sym_mat<prec_type>(q(all, cidx), mat_size) * pinv, mat_size);
            }
        }
        return out;
    }
    /// @brief Computes the inverse hessian-vector product (IHVP) F''(p)^-1 q.
    ///
    /// For the PSD cone, this is given by vec(mat(p) mat(q) mat(p)).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the inverse Hessian.
    /// @param q Vector to take the IHVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    static optMatrix<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& q, const PSDParameters& cone_params){
        using namespace Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto p = vec_to_her_mat<prec_type>(p, mat_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(p * vec_to_her_mat<prec_type>(q(all, cidx)) * p, mat_size);
            }
        }else{
            auto p = vec_to_sym_mat<prec_type>(p, mat_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec<prec_type>(p * vec_to_sym_mat<prec_type>(q(all, cidx)) * p, mat_size);
            }
        }
        return out;
    }
};

#endif