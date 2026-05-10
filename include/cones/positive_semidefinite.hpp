#ifndef POSITIVE_SEMIDEFINITE_CONES_H
#define POSITIVE_SEMIDEFINITE_CONES_H
#include "cone.hpp"
#include "common_typedefs.hpp"
#include <Eigen/Cholesky>

class PositiveSemidefiniteParameters : public ConeParameters{
protected:
    int matrix_size;
public:
    void parse_args(const std::string& args) override;
    int matrixSize(){return matrix_size;}
};

template<typename prec_type, bool complex_type>
class PositiveSemidefinite : public Cone{
protected:
    static const bool is_symmetric = true;
    static const bool is_complex = complex_type;
public:
    static bool isSymmetric() const{return is_symmetric;}
    static bool isComplex() const{return is_complex;}
    static std::string coneId() const{return std::string("PSD");}
    static std::string coneName() const{
        return (is_complex ? std::string("Complex positive semi-definite cone") :
        std::string("Real positive semi-definite cone"));
    }
    /// @brief Computes the gradient F'(p).
    ///
    /// For the PSD cone this is given by -vec(mat(p)^-1).
    /// @param p Point at which to calculate the gradient.
    template<typename Derived>
    static optVector<prec_type> grad(const Eigen::MatrixBase<Derived>& p, const std::unique_ptr<ConeParameters>& cone_params){
        int mat_size = cone_params->matrixSize();
        if(is_complex)  return -her_mat_to_vec(vec_to_her_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size)));
        return -sym_mat_to_vec(vec_to_sym_mat(p, mat_size).ldlt().solve(optMatrix::Identity(mat_size, mat_size)));
    }
    /// @brief Computes the hessian-vector product (HVP) F''(p)q.
    ///
    /// For the PSD cone, this is given by vec(mat(p)^-1 mat(q) mat(p)^-1).
    /// If q is a matrix, this returns a matrix [c1 c2 c3 ...] where the columns are the results.
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the HVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    template<typename Derived>
    static optMatrix<prec_type> hvp(const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q, const std::unique_ptr<ConeParameters>& cone_params){
        using Eigen::placeholders;
        int mat_size = cone_params->matrixSize();
        optMatrix<prec_type> out;
        out.resizeLike(q);
        if(is_complex){
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
    /// @param p Point at which to calculate the Hessian.
    /// @param q Vector to take the IHVP with. q may be a matrix, which is interpreted as a series
    /// of column vectors.
    template<typename Derived>
    static optMatrix<prec_type> ihvp(const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& q, const std::unique_ptr<ConeParameters>& cone_params){
        using Eigen::placeholders;
        int mat_size = cone_params->matrixSize();
        optMatrix<prec_type> out;
        out.resizeLike(q);
        if(is_complex){
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
};

#endif