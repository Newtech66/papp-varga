#ifndef PSD_CONES_H
#define PSD_CONES_H
#include "psd_parameters.hpp"
#include "common_typedefs.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

template<typename prec_type, bool is_complex>
class PSD{
private:
    // Singular value decompositions
    static Eigen::BDCSVD<prec_type, Eigen::ComputeThinV> sym_svd;
    static Eigen::BDCSVD<std::complex<prec_type>, Eigen::ComputeThinV> her_svd;

    // Eigenvalue decompositions
    static Eigen::SelfAdjointEigenSolver<prec_type> sym_eigsolver;
    static Eigen::SelfAdjointEigenSolver<std::complex<prec_type>> her_eigsolver;
public:
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    template<typename T>
    static optVector<prec_type> grad(const Eigen::MatrixBase<T>& p, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            return -her_mat_to_vec(vec_to_her_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size)));
        }else{
            return -sym_mat_to_vec(vec_to_sym_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size)));
        }
    }
    /// @brief Computes the hessian-vector product (HVP) F''(p)q.
    ///
    /// For the PSD cone, this is given by vec(mat(p)^-1 mat(q) mat(p)^-1).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the HVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    template<typename T, typename U>
    static optMatrix<prec_type> hvp(const Eigen::MatrixBase<T>& p, const Eigen::MatrixBase<U>& q, const PSDParameters& cone_params){
        using Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto pinv = vec_to_her_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size));
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec(pinv * vec_to_her_mat(q(all, cidx)) * pinv);
            }
        }else{
            auto pinv = vec_to_sym_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size));
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec(pinv * vec_to_sym_mat(q(all, cidx)) * pinv);
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
    template<typename T, typename U>
    static optMatrix<prec_type> ihvp(const Eigen::MatrixBase<T>& p, const Eigen::MatrixBase<U>& q, const PSDParameters& cone_params){
        using Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto p = vec_to_her_mat(p, mat_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec(p * vec_to_her_mat(q(all, cidx)) * p);
            }
        }else{
            auto p = vec_to_sym_mat(p, mat_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec(p * vec_to_sym_mat(q(all, cidx)) * p);
            }
        }
        return out;
    }
    template<typename T, typename U>
    static void get_nt_scaling(const optVector<prec_type>& s, const optVector<prec_type>& z,
        optMatrix<prec_type>& scaling_matrix, Eigen::MatrixBase<T>& scaling_point, Eigen::MatrixBase<U>& scaled_variable,
        const PSDParameters& cone_params){
        if constexpr(is_complex){
            auto S = vec_to_her_mat(s);
            auto Z = vec_to_her_mat(z);
            auto L1 = S.llt().matrixL();
            auto L2 = Z.llt().matrixL();
            her_svd.compute(L2.conj() * L1);
            scaling_matrix = L1 * her_svd.matrixV() * her_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = her_mat_to_vec(her_svd.singularValues().asDiagonal());
            scaling_point = her_mat_to_vec(scaling_matrix * scaling_matrix.transpose());
        }else{
            auto S = vec_to_sym_mat(s);
            auto Z = vec_to_sym_mat(z);
            auto L1 = S.llt().matrixL();
            auto L2 = Z.llt().matrixL();
            sym_svd.compute(L2.conj() * L1);
            scaling_matrix = L1 * sym_svd.matrixV() * sym_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = sym_mat_to_vec(sym_svd.singularValues().asDiagonal());
            scaling_point = sym_mat_to_vec(scaling_matrix * scaling_matrix.transpose());
        }
    }
    template<typename T, typename U>
    static void get_nt_step_length(const optVector<prec_type>& s, const optVector<prec_type>& z, Eigen::MatrixBase<U>& scaled_variable){
        if constexpr(is_complex){
            optMatrix<prec_type> lis = vec_to_her_mat(scaled_variable).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_her_mat(s) * lis;
            auto sigk = lis * vec_to_her_mat(z) * lis;
            her_eigsolver.compute(rhok);
            prec_type gams = her_eigsolver.eigenvalues()(0);
            her_eigsolver.compute(sigk);
            prec_type gamz = her_eigsolver.eigenvalues()(0);
            alphak.push_back(1 / std::max({0, -gams, -gamz}));
        }else{
            optMatrix<prec_type> lis = vec_to_sym_mat(scaled_variable).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_sym_mat(s) * lis;
            auto sigk = lis * vec_to_sym_mat(z) * lis;
            sym_eigsolver.compute(rhok);
            prec_type gams = sym_eigsolver.eigenvalues()(0);
            sym_eigsolver.compute(sigk);
            prec_type gamz = sym_eigsolver.eigenvalues()(0);
            alphak.push_back(1 / std::max({0, -gams, -gamz}));
        }
    }
    // static optVector<prec_type> circle_product(const Eigen::MatrixBase<T>& u, const Eigen::MatrixBase<U>& v, const PSDParameters& cone_params){
    //     if constexpr(is_complex){
    //         auto umat = vec_to_sym_mat(u);
    //         auto vmat = vec_to_sym_mat(v);
    //         return sym_mat_to_vec(umat * vmat + vmat * umat) / prec_type(2);
    //     }else{
    //         auto umat = vec_to_her_mat(u);
    //         auto vmat = vec_to_her_mat(v);
    //         return her_mat_to_vec(umat * vmat + vmat * umat) / prec_type(2);
    //     }
    // }
    // static optVector<prec_type> diamond_product(const Eigen::MatrixBase<T>& u, const Eigen::MatrixBase<U>& v, const PSDParameters& cone_params){
    //     if constexpr(is_complex){
    //         auto umat = vec_to_sym_mat(u);
    //         auto vmat = vec_to_sym_mat(v);
    //         return sym_mat_to_vec(umat * vmat + vmat * umat) / prec_type(2);
    //     }else{
    //         auto umat = vec_to_her_mat(u);
    //         auto vmat = vec_to_her_mat(v);
    //         return her_mat_to_vec(umat * vmat + vmat * umat) / prec_type(2);
    //     }
    // }
};

#endif