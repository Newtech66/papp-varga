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
    Eigen::LDLT<optMatrix<wtype>> ldlt;

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
            optMatrix<wtype> pinv = vec_to_her_mat<prec_type>(p, matrix_size).ldlt().solve(iden);
            for(int cidx = 0; cidx < q.cols(); cidx++){
                out(all, cidx) = her_mat_to_vec<prec_type>(pinv * vec_to_her_mat<prec_type>(q(all, cidx), matrix_size) * pinv, matrix_size);
            }
        }else{
            optMatrix<wtype> pinv = vec_to_sym_mat<prec_type>(p, matrix_size).ldlt().solve(iden);
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
            optMatrix<wtype> L1 = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            L1.template triangularView<Eigen::Lower>() = S.llt().matrixL();
            svd.compute(Z.llt().solve(Z * L1));
            lam = svd.singularValues();
            R = L1 * svd.matrixV() * lam.cwiseSqrt().cwiseInverse().asDiagonal();
            w = her_mat_to_vec<prec_type>(R * R.transpose(), matrix_size);
        }else{
            auto S = vec_to_sym_mat<prec_type>(s, matrix_size);
            auto Z = vec_to_sym_mat<prec_type>(z, matrix_size);
            optMatrix<wtype> L1 = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            L1.template triangularView<Eigen::Lower>() = S.llt().matrixL();
            svd.compute(Z.llt().solve(Z * L1));
            lam = svd.singularValues();
            R = L1 * svd.matrixV() * lam.cwiseSqrt().cwiseInverse().asDiagonal();
            w = sym_mat_to_vec<prec_type>(R * R.transpose(), matrix_size);
        }
    }
    prec_type get_nt_step_length(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z){
        if constexpr(is_complex){
            auto lam_isqrt_diag = lam.cwiseInverse().cwiseSqrt().asDiagonal();
            optMatrix<wtype> rhok = lam_isqrt_diag * vec_to_her_mat<prec_type>(s, matrix_size) * lam_isqrt_diag;
            optMatrix<wtype> sigk = lam_isqrt_diag * vec_to_her_mat<prec_type>(z, matrix_size) * lam_isqrt_diag;
            eigsolver.compute(rhok);
            prec_type gams = eigsolver.eigenvalues()(0);
            eigsolver.compute(sigk);
            prec_type gamz = eigsolver.eigenvalues()(0);
            return prec_type(1) / std::max({prec_type(0), -gams, -gamz});
        }else{
            auto lam_isqrt_diag = lam.cwiseInverse().cwiseSqrt().asDiagonal();
            optMatrix<wtype> rhok = lam_isqrt_diag * vec_to_sym_mat<prec_type>(s, matrix_size) * lam_isqrt_diag;
            optMatrix<wtype> sigk = lam_isqrt_diag * vec_to_sym_mat<prec_type>(z, matrix_size) * lam_isqrt_diag;
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
        // W^-1.T ds o W dz = 1/2 vec(mat(W^-1.T ds) mat(W dz) + mat(W dz) mat(W^-1.T ds))
        //                  = 1/2 vec(R^-1 mat(ds) R^-T R^T mat(dz) R + R^T mat(dz) R R^-1 mat(ds) R^-T)
        //                  = 1/2 vec(R^-1 mat(ds) mat(dz) R + R^T mat(dz) mat(ds) R^-T)
        // l @ u = vec(mat(u) $ Gam)
        // Gam = 2 / (Lam_ii + Lam_jj)
        // R^-T mat(u) R^-1 = R (R^T R)^-1 mat(u) R^-1 = R mat(w)^-1 mat(u) R^-1 = R M R^-1
        //                  = R (R^-T M^T)^T = R (R mat(w)^-1 M^T)^T = R (R P)^T = R P^T R^T
        // or,
        // R^-T U R^-1 = M R^-1 = (R^-T M)^T = P^T
        // the first method uses 2 ldlt solves and the second method uses 2 qr solves
        if constexpr(is_complex){
            qr.compute(R);
            optMatrix<wtype> Q = qr.solve(vec_to_her_mat<prec_type>(s, matrix_size) * vec_to_her_mat<prec_type>(z, matrix_size)) * R;
            Q += Q.conjugate().eval();
            optMatrix<prec_type> Gam = lam.replicate(1, lam.rows());
            Gam += Gam.transpose().eval();
            Q = Q.cwiseQuotient(Gam).eval();
            optMatrix<wtype> Lam = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            optMatrix<wtype> Lami = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            Lam.diagonal() = lam;
            Lami.diagonal() = lam.cwiseInverse();
            optMatrix<wtype> U = -Lam + centering_parameter * mu * Lami - Q;
            ldlt.compute(vec_to_her_mat<prec_type>(w, matrix_size));
            optMatrix<wtype> M = ldlt.solve(U);
            optMatrix<wtype> P = ldlt.solve(M.conjugate());
            return her_mat_to_vec<prec_type>(R * P.conjugate() * R.conjugate(), matrix_size);
        }else{
            qr.compute(R);
            optMatrix<wtype> Q = qr.solve(vec_to_sym_mat<prec_type>(s, matrix_size) * vec_to_sym_mat<prec_type>(z, matrix_size)) * R;
            Q += Q.conjugate().eval();
            optMatrix<prec_type> Gam = lam.replicate(1, lam.rows());
            Gam += Gam.transpose().eval();
            Q = Q.cwiseQuotient(Gam).eval();
            optMatrix<wtype> Lam = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            optMatrix<wtype> Lami = optMatrix<wtype>::Zero(matrix_size, matrix_size);
            Lam.diagonal() = lam;
            Lami.diagonal() = lam.cwiseInverse();
            optMatrix<wtype> U = -Lam + centering_parameter * mu * Lami - Q;
            ldlt.compute(vec_to_sym_mat<prec_type>(w, matrix_size));
            optMatrix<wtype> M = ldlt.solve(U);
            optMatrix<wtype> P = ldlt.solve(M.conjugate());
            return sym_mat_to_vec<prec_type>(R * P.conjugate() * R.conjugate(), matrix_size);
        }
    }
    optVector<prec_type> get_nt_scaling_point(){
        if constexpr(is_complex){
            optMatrix<std::complex<prec_type>> tmp(matrix_size, matrix_size);
            tmp.setZero();
            tmp.diagonal() = lam;
            return her_mat_to_vec<prec_type>(tmp, matrix_size);
        }
        else{
            optMatrix<prec_type> tmp(matrix_size, matrix_size);
            tmp.setZero();
            tmp.diagonal() = lam;
            return sym_mat_to_vec<prec_type>(tmp, matrix_size);
        }
    }
    optVector<prec_type> get_nt_scaled_variable(){
        return w;
    }
};

#endif