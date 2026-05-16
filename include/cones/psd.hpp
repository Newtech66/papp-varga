#ifndef PSD_CONES_H
#define PSD_CONES_H
#include <string>
#include <format>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include "common_typedefs.hpp"
#include "mat_vector_transforms.hpp"
#include <stdexcept>

template<typename prec_type, bool is_complex>
class PSD{
    using wtype = std::conditional_t<is_complex, std::complex<prec_type>, prec_type>;
protected:
    static const bool is_symmetric = true;
    const int matrix_size;

    // Nesterov-Todd specific stuff
    // the matrix R, Wv = R.T mat(v) R
    optMatrix<wtype> R, iden;
    // The scaling point w
    // lam, which is related to the scaled variable Lam by Lam = vec(diag(lam))
    optVector<prec_type> w, lam;
    // Decompositions
    Eigen::BDCSVD<optMatrix<wtype>, Eigen::ComputeThinV> svd;
    Eigen::SelfAdjointEigenSolver<optMatrix<wtype>> eigsolver;
    Eigen::ColPivHouseholderQR<optMatrix<wtype>> qr;

public:
    PSD(int matrix_size): matrix_size(matrix_size){
        // size up R, w, and lam here
        iden = optMatrix<wtype>::Identity(matrix_size, matrix_size);
        R.resize(matrix_size, matrix_size);
        w.resize(numVariables());
        lam.resize(matrix_size);
    }
    static bool isSymmetric(){return is_symmetric;}
    int barrierParameter() const{return matrix_size;}
    int numVariables() const{
        if constexpr(is_complex)  return matrix_size * (matrix_size + 1);
        else    return matrix_size * (matrix_size + 1) / 2;
    }
    std::string coneName() const{
        if constexpr(is_complex){
            return std::format("Cone of {0} x {0} complex Hermitian positive-semidefinite matrices", matrix_size);
        }else{
            return std::format("Cone of {0} x {0} real symmetric positive-semidefinite matrices", matrix_size);
        }
    }
    bool isComplex() const{return is_complex;}
    int matrixSize() const{return matrix_size;}
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    optVector<prec_type> grad(const Eigen::Ref<const optVector<prec_type>>& p){
        if constexpr(is_complex){
            return -her_mat_to_vec<prec_type>(vec_to_her_mat<prec_type>(p, matrix_size).ldlt().solve(iden), matrix_size);
        }else{
            return -sym_mat_to_vec<prec_type>(vec_to_sym_mat<prec_type>(p, matrix_size).ldlt().solve(iden), matrix_size);
        }
    }
    /// @brief Computes the hessian-vector product (HVP) F''(p)q.
    ///
    /// For the PSD cone, this is given by vec(mat(p)^-1 mat(q) mat(p)^-1).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the HVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    optMatrix<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& q){
        using namespace Eigen::placeholders;
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto pinv = vec_to_her_mat<prec_type>(p, matrix_size).ldlt().solve(iden);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(pinv * vec_to_her_mat<prec_type>(q(all, cidx), matrix_size) * pinv, matrix_size);
            }
        }else{
            auto pinv = vec_to_sym_mat<prec_type>(p, matrix_size).ldlt().solve(iden);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec<prec_type>(pinv * vec_to_sym_mat<prec_type>(q(all, cidx), matrix_size) * pinv, matrix_size);
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
    optMatrix<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& q){
        using namespace Eigen::placeholders;
        optMatrix<prec_type> out;
        out.resize(q.rows(), q.cols());
        if constexpr(is_complex){
            auto p = vec_to_her_mat<prec_type>(p, matrix_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(p * vec_to_her_mat<prec_type>(q(all, cidx)) * p, matrix_size);
            }
        }else{
            auto p = vec_to_sym_mat<prec_type>(p, matrix_size);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = sym_mat_to_vec<prec_type>(p * vec_to_sym_mat<prec_type>(q(all, cidx)) * p, matrix_size);
            }
        }
        return out;
    }
    void update_nt_scaling(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z){
        if constexpr(is_complex){
            auto S = vec_to_her_mat<prec_type>(s, matrix_size);
            auto Z = vec_to_her_mat<prec_type>(z, matrix_size);
            auto L1 = S.llt().matrixL();
            svd.compute(Z.llt().solve(Z * L1));
            lam = svd.singularValues();
            R = L1 * svd.matrixV() * lam.cwiseSqrt().cwiseInverse().asDiagonal();
            w = her_mat_to_vec<prec_type>(R * R.transpose(), matrix_size);
        }else{
            auto S = vec_to_sym_mat<prec_type>(s, matrix_size);
            auto Z = vec_to_sym_mat<prec_type>(z, matrix_size);
            auto L1 = S.llt().matrixL();
            svd.compute(Z.llt().solve(Z * L1));
            lam = svd.singularValues();
            R = L1 * svd.matrixV() * lam.cwiseSqrt().cwiseInverse().asDiagonal();
            w = sym_mat_to_vec<prec_type>(R * R.transpose(), matrix_size);
        }
    }
    prec_type get_nt_step_length(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z){
        if constexpr(is_complex){
            auto lam_isqrt_diag = lam.cwiseInverse().cwiseSqrt().asDiagonal();
            auto rhok = lam_isqrt_diag * vec_to_her_mat<prec_type>(s, matrix_size) * lam_isqrt_diag;
            auto sigk = lam_isqrt_diag * vec_to_her_mat<prec_type>(z, matrix_size) * lam_isqrt_diag;
            eigsolver.compute(rhok);
            prec_type gams = eigsolver.eigenvalues()(0);
            eigsolver.compute(sigk);
            prec_type gamz = eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({prec_type(0), -gams, -gamz});
        }else{
            auto lam_isqrt_diag = lam.cwiseInverse().cwiseSqrt().asDiagonal();
            auto rhok = lam_isqrt_diag * vec_to_sym_mat<prec_type>(s, matrix_size) * lam_isqrt_diag;
            auto sigk = lam_isqrt_diag * vec_to_sym_mat<prec_type>(z, matrix_size) * lam_isqrt_diag;
            eigsolver.compute(rhok);
            prec_type gams = eigsolver.eigenvalues()(0);
            eigsolver.compute(sigk);
            prec_type gamz = eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({prec_type(0), -gams, -gamz});
        }
    }
    optVector<prec_type> get_nt_rhs_s(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z, const prec_type centering_parameter, const prec_type mu){
        //W^-1 (l @ (-l o l - (W^-1.T ds) o (W dz) + sig mu E))
        // for this cone, W^-1(-l + sig mu l^-1 - l @ (W^-1.T ds o W dz))
        // remember that W^-T s = vec(R^-1 mat(s) R^-T) and Wz = vec(R^T mat(z) R)
        // similarly W^-1 u = vec(R^-T mat(u) R^-1) (according to QICS anyways)
        throw std::logic_error("No implementation of get_nt_rhs_s for positive-semidefinite cone!");
    }
};

#endif