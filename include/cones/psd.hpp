#ifndef PSD_CONES_H
#define PSD_CONES_H
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include "psd_parameters.hpp"
#include "common_typedefs.hpp"
#include "mat_vector_transforms.hpp"

template<typename prec_type, bool is_complex>
class PSD{
private:
    // Singular value decompositions
    static Eigen::BDCSVD<optMatrix<prec_type>, Eigen::ComputeThinV> sym_svd;
    static Eigen::BDCSVD<optMatrix<std::complex<prec_type>>, Eigen::ComputeThinV> her_svd;

    // Eigenvalue decompositions
    static Eigen::SelfAdjointEigenSolver<optMatrix<prec_type>> sym_eigsolver;
    static Eigen::SelfAdjointEigenSolver<optMatrix<std::complex<prec_type>>> her_eigsolver;

    // QR decompositions
    static Eigen::ColPivHouseholderQR<optMatrix<prec_type>> sym_qr;
    static Eigen::ColPivHouseholderQR<optMatrix<std::complex<prec_type>>> her_qr;

    // LLT decompositions
    static Eigen::ColPivHouseholderQR<optMatrix<prec_type>> sym_llt;
    static Eigen::ColPivHouseholderQR<optMatrix<std::complex<prec_type>>> her_llt;
public:
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    static optVector<prec_type> grad(const Eigen::Ref<const optVector<prec_type>>& p, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            return -her_mat_to_vec<prec_type>(vec_to_her_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size)), mat_size);
        }else{
            return -sym_mat_to_vec<prec_type>(vec_to_sym_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size)), mat_size);
        }
    }
    /// @brief Computes the hessian-vector product (HVP) F''(p)q.
    ///
    /// For the PSD cone, this is given by vec(mat(p)^-1 mat(q) mat(p)^-1).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the HVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    static optMatrix<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optVector<prec_type>>& q, const PSDParameters& cone_params){
        using namespace Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto pinv = vec_to_her_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size));
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(pinv * vec_to_her_mat<prec_type>(q(all, cidx), mat_size) * pinv, mat_size);
            }
        }else{
            auto pinv = vec_to_sym_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size));
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
    static optMatrix<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optVector<prec_type>>& q, const PSDParameters& cone_params){
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
    static void get_nt_scaling(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z, Eigen::Ref<optMatrix<prec_type>> scaling_matrix, Eigen::Ref<optVector<prec_type>> scaling_point, Eigen::Ref<optVector<prec_type>> scaled_variable, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            auto S = vec_to_her_mat<prec_type>(s, mat_size);
            auto Z = vec_to_her_mat<prec_type>(z, mat_size);
            auto L1 = S.llt().matrixL();
            her_llt.compute(Z);
            her_svd.compute(her_llt.solve(Z * L1));
            optMatrix<prec_type> lam = her_svd.singularValues().asDiagonal();
            scaling_matrix = L1 * her_svd.matrixV() * her_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = her_mat_to_vec<prec_type>(lam, mat_size);
            scaling_point = her_mat_to_vec<prec_type>(scaling_matrix * scaling_matrix.transpose(), mat_size);
        }else{
            auto S = vec_to_sym_mat<prec_type>(s, mat_size);
            auto Z = vec_to_sym_mat<prec_type>(z, mat_size);
            auto L1 = S.llt().matrixL();
            sym_llt.compute(Z);
            sym_svd.compute(sym_llt.solve(Z * L1));
            optMatrix<prec_type> lam = sym_svd.singularValues().asDiagonal();
            scaling_matrix = L1 * sym_svd.matrixV() * sym_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = sym_mat_to_vec<prec_type>(lam, mat_size);
            scaling_point = sym_mat_to_vec<prec_type>(scaling_matrix * scaling_matrix.transpose(), mat_size);
        }
    }
    static prec_type get_nt_step_length(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z, const Eigen::Ref<const optVector<prec_type>>& scaled_variable, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            auto lis = vec_to_her_mat<prec_type>(scaled_variable, mat_size).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_her_mat<prec_type>(s, mat_size) * lis;
            auto sigk = lis * vec_to_her_mat<prec_type>(z, mat_size) * lis;
            her_eigsolver.compute(rhok);
            prec_type gams = her_eigsolver.eigenvalues()(0);
            her_eigsolver.compute(sigk);
            prec_type gamz = her_eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({prec_type(0), -gams, -gamz});
        }else{
            auto lis = vec_to_sym_mat<prec_type>(scaled_variable, mat_size).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_sym_mat<prec_type>(s, mat_size) * lis;
            auto sigk = lis * vec_to_sym_mat<prec_type>(z, mat_size) * lis;
            sym_eigsolver.compute(rhok);
            prec_type gams = sym_eigsolver.eigenvalues()(0);
            sym_eigsolver.compute(sigk);
            prec_type gamz = sym_eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({prec_type(0), -gams, -gamz});
        }
    }
    static optVector<prec_type> get_nt_rhs_s(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z, const Eigen::Ref<const optMatrix<prec_type>>& scaling_matrix, const Eigen::Ref<const optVector<prec_type>>& scaled_variable, const prec_type centering_parameter, const prec_type mu, const PSDParameters& cone_params){
        //W.T (l @ (-l o l - (W^-1.T ds) o (W dz) + sig mu E))
        // for this cone, W.T(-l + sig mu l^-1 - l @ (W^-1.T ds o W dz))
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            sym_qr.compute(scaling_matrix.transpose());
            optMatrix<prec_type> Wtis = vec_to_her_mat<prec_type>(sym_qr.solve(s), mat_size);
            optMatrix<prec_type> Wz = vec_to_her_mat<prec_type>(scaling_matrix * z, mat_size);
            // V_ij = (Wtis * Wz + Wz * Wtis)_ij / (L_ii + L_jj)
            optMatrix<prec_type> gam = scaled_variable.replicate(1, scaled_variable.rows());
            gam += gam.transpose();
            optMatrix<prec_type> lami = scaled_variable.diagonal().cwiseInverse().asDiagonal();
            return scaling_matrix.transpose() * (-scaled_variable +
                centering_parameter * mu * her_mat_to_vec<prec_type>(lami, mat_size) -
                her_mat_to_vec<prec_type>((Wtis * Wz + Wz * Wtis).cwiseQuotient(gam), mat_size));
        }else{
            sym_qr.compute(scaling_matrix.transpose());
            optMatrix<prec_type> Wtis = vec_to_sym_mat<prec_type>(sym_qr.solve(s), mat_size);
            optMatrix<prec_type> Wz = vec_to_sym_mat<prec_type>(scaling_matrix * z, mat_size);
            // V_ij = (Wtis * Wz + Wz * Wtis)_ij / (L_ii + L_jj)
            optMatrix<prec_type> gam = scaled_variable.replicate(1, scaled_variable.rows());
            gam += gam.transpose();
            optMatrix<prec_type> lami = scaled_variable.diagonal().cwiseInverse().asDiagonal();
            return scaling_matrix.transpose() * (-scaled_variable +
                centering_parameter * mu * sym_mat_to_vec<prec_type>(lami, mat_size) -
                sym_mat_to_vec<prec_type>((Wtis * Wz + Wz * Wtis).cwiseQuotient(gam), mat_size));
        }
    }
};

#endif