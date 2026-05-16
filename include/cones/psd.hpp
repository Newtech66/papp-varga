#ifndef PSD_CONES_H
#define PSD_CONES_H
#include "psd_parameters.hpp"
#include "common_typedefs.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

template<typename prec_type, bool is_complex>
class PSD{
private:
    // Singular value decompositions
    static Eigen::BDCSVD<prec_type, Eigen::ComputeThinV> sym_svd;
    static Eigen::BDCSVD<std::complex<prec_type>, Eigen::ComputeThinV> her_svd;

    // Eigenvalue decompositions
    static Eigen::SelfAdjointEigenSolver<prec_type> sym_eigsolver;
    static Eigen::SelfAdjointEigenSolver<std::complex<prec_type>> her_eigsolver;

    // QR decompositions
    static Eigen::ColPivHouseholderQR<prec_type> sym_qr;
    static Eigen::ColPivHouseholderQR<std::complex<prec_type>> her_qr;
public:
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    static optVector<prec_type> grad(const optVector<prec_type>& p, const PSDParameters& cone_params){
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
    static optMatrix<prec_type> hvp(const optVector<prec_type>& p, const optVector<prec_type>& q, const PSDParameters& cone_params){
        using namespace Eigen::placeholders;
        int mat_size = cone_params.matrixSize();
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto pinv = vec_to_her_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size), mat_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(pinv * vec_to_her_mat<prec_type>(q(all, cidx), mat_size) * pinv, mat_size);
            }
        }else{
            auto pinv = vec_to_sym_mat<prec_type>(p, mat_size).ldlt().solve(optMatrix<prec_type>::Identity(mat_size, mat_size), mat_size);
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
    static optMatrix<prec_type> ihvp(const optVector<prec_type>& p, const optVector<prec_type>& q, const PSDParameters& cone_params){
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
    static void get_nt_scaling(const optVector<prec_type>& s, const optVector<prec_type>& z, optMatrix<prec_type>& scaling_matrix, optVector<prec_type>& scaling_point, optVector<prec_type>& scaled_variable, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            auto S = vec_to_her_mat<prec_type>(s, mat_size);
            auto Z = vec_to_her_mat<prec_type>(z, mat_size);
            auto L1 = S.llt().matrixL();
            auto L2 = Z.llt().matrixL();
            her_svd.compute(L2.conj() * L1);
            scaling_matrix = L1 * her_svd.matrixV() * her_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = her_mat_to_vec<prec_type>(her_svd.singularValues().asDiagonal(), mat_size);
            scaling_point = her_mat_to_vec<prec_type>(scaling_matrix * scaling_matrix.transpose(), mat_size);
        }else{
            auto S = vec_to_sym_mat<prec_type>(s, mat_size);
            auto Z = vec_to_sym_mat<prec_type>(z, mat_size);
            auto L1 = S.llt().matrixL();
            auto L2 = Z.llt().matrixL();
            sym_svd.compute(L2.conj() * L1);
            scaling_matrix = L1 * sym_svd.matrixV() * sym_svd.singularValues().cwiseSqrt().cwiseInverse().asDiagonal();
            scaled_variable = sym_mat_to_vec<prec_type>(sym_svd.singularValues().asDiagonal(), mat_size);
            scaling_point = sym_mat_to_vec<prec_type>(scaling_matrix * scaling_matrix.transpose(), mat_size);
        }
    }
    static void get_nt_step_length(const optVector<prec_type>& s, const optVector<prec_type>& z, optVector<prec_type>& scaled_variable, const PSDParameters& cone_params){
        int mat_size = cone_params.matrixSize();
        if constexpr(is_complex){
            optMatrix<prec_type> lis = vec_to_her_mat<prec_type>(scaled_variable, mat_size).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_her_mat<prec_type>(s, mat_size) * lis;
            auto sigk = lis * vec_to_her_mat<prec_type>(z, mat_size) * lis;
            her_eigsolver.compute(rhok);
            prec_type gams = her_eigsolver.eigenvalues()(0);
            her_eigsolver.compute(sigk);
            prec_type gamz = her_eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({0, -gams, -gamz});
        }else{
            optMatrix<prec_type> lis = vec_to_sym_mat<prec_type>(scaled_variable, mat_size).cwiseInverse().cwiseSqrt();
            auto rhok = lis * vec_to_sym_mat<prec_type>(s, mat_size) * lis;
            auto sigk = lis * vec_to_sym_mat<prec_type>(z, mat_size) * lis;
            sym_eigsolver.compute(rhok);
            prec_type gams = sym_eigsolver.eigenvalues()(0);
            sym_eigsolver.compute(sigk);
            prec_type gamz = sym_eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({0, -gams, -gamz});
        }
    }
    static optVector<prec_type> get_nt_rhs_s(const optVector<prec_type>& s, const optVector<prec_type>& z, const optMatrix<prec_type> scaling_matrix, const optVector<prec_type>& scaled_variable, const prec_type centering_parameter, const prec_type mu){
        //W.T (l @ (-l o l - (W^-1.T ds) o (W dz) + sig mu E))
        // for this cone, W.T(-l + sig mu l^-1 - l @ (W^-1.T ds o W dz))
        if constexpr(is_complex){
            auto lam = vec_to_her_mat<prec_type>(scaled_variable);
            her_qr.compute(scaling_matrix.transpose());
            optMatrix<prec_type> Wtis = vec_to_her_mat<prec_type>(her_qr.solve(s));
            optMatrix<prec_type> Wz = vec_to_her_mat<prec_type>(scaling_matrix * z);
            // V_ij = (Wtis * Wz + Wz * Wtis)_ij / (L_ii + L_jj)
            optMatrix<prec_type> gam = scaled_variable.replicate(scaled_variable.rows());
            gam += gam.transpose();
            return scaling_matrix.transpose() * (-scaled_variable +
                centering_parameter * mu * her_mat_to_vec<prec_type>(scaled_variable.diagonal().cwiseInverse().asDiagonal()) -
                her_mat_to_vec<prec_type>((Wtis * Wz + Wz * Wtis).cwiseQuotient(gam)));
        }else{
            auto lam = vec_to_sym_mat<prec_type>(scaled_variable);
            sym_qr.compute(scaling_matrix.transpose());
            optMatrix<prec_type> Wtis = vec_to_sym_mat<prec_type>(sym_qr.solve(s));
            optMatrix<prec_type> Wz = vec_to_sym_mat<prec_type>(scaling_matrix * z);
            // V_ij = (Wtis * Wz + Wz * Wtis)_ij / (L_ii + L_jj)
            optMatrix<prec_type> gam = scaled_variable.replicate(scaled_variable.rows());
            gam += gam.transpose();
            return scaling_matrix.transpose() * (-scaled_variable +
                centering_parameter * mu * sym_mat_to_vec<prec_type>(scaled_variable.diagonal().cwiseInverse().asDiagonal()) -
                sym_mat_to_vec<prec_type>((Wtis * Wz + Wz * Wtis).cwiseQuotient(gam)));
        }
    }
};

#endif