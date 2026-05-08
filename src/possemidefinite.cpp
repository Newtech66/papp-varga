#ifndef POSSEMIDEFINITE_PAPP_VARGA_H
#define POSSEMIDEFINITE_PAPP_VARGA_H
#include "vectorize.hpp"
#include "cones.hpp"
#include "common_typedefs.hpp"
#include "complex_deduction.hpp"
#include <type_traits>
#include <Eigen/LU>

template<typename prec_type>
class RealPositiveSemidefinite : public Cone<prec_type>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return type is RealVector
protected:
    int matrix_size;
    optMatrix<prec_type> P, Pinv, iden;
    bool jac_updated = false;
public:
    RealPositiveSemidefinite(const int n) : matrix_size(n){
        iden.setIdentity(matrix_size, matrix_size);
        P.setIdentity(matrix_size, matrix_size);
        this->barrier_parameter = matrix_size;
        this->num_params = matrix_size * matrix_size;
    }
    std::string coneName() const override{return std::string("Real positive semi-definite cone");}
    optVector<prec_type> point() const override{return vec<prec_type>(P);}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override{
        P = unvecReal<prec_type>(p, matrix_size);
        jac_updated = false;
    }
    optVector<prec_type> jacobian() override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(iden);
            jac_updated = true;
        }
        return -vec<prec_type>(Pinv);
    }
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(iden);
            jac_updated = true;
        }
        return vec<prec_type>(Pinv * unvecReal<prec_type>(v, matrix_size) * Pinv);
    }
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return vec<prec_type>(P * unvecReal<prec_type>(v, matrix_size) * P);
    }
};

template<typename prec_type>
class ComplexPositiveSemidefinite : public Cone<prec_type>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return type is RealVector
protected:
    int matrix_size;
    optMatrix<std::complex<prec_type>> P, Pinv, iden;
    bool jac_updated = false;
public:
    ComplexPositiveSemidefinite(const int n) : matrix_size(n){
        iden.setIdentity(matrix_size, matrix_size);
        P.setIdentity(matrix_size, matrix_size);
        this->barrier_parameter = matrix_size;
        this->num_params = 2 * matrix_size * matrix_size;
    }
    std::string coneName() const override{return std::string("Complex positive semi-definite cone");}
    optVector<prec_type> point() const override{return vec<prec_type>(P);}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override{
        P = unvecComplex<prec_type>(p, matrix_size);
        jac_updated = false;
    }
    optVector<prec_type> jacobian() override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(iden);
            jac_updated = true;
        }
        return -vec<prec_type>(Pinv);
    }
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(iden);
            jac_updated = true;
        }
        return vec<prec_type>(Pinv * unvecComplex<prec_type>(v, matrix_size) * Pinv);
    }
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return vec<prec_type>(P * unvecComplex<prec_type>(v, matrix_size) * P);
    }
};

template<typename prec_type>
class DiagonalPositiveSemidefinite : public Cone<prec_type>{
protected:
    int matrix_size;
    optVector<prec_type> p, pinv;
public:
    DiagonalPositiveSemidefinite(const int n) : matrix_size(n){
        p.setOnes(matrix_size);
        pinv.setOnes(matrix_size);
        this->barrier_parameter = matrix_size;
        this->num_params = matrix_size;
    }
    std::string coneName() const override{return std::string("Diagonal real positive semi-definite cone");}
    optVector<prec_type> point() const override{return p;}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override{
        this->p = p;
        pinv = p.inverse();
    }
    optVector<prec_type> jacobian() override{return -pinv;}
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return pinv.cwiseProduct(v).cwiseProduct(pinv);
    }
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return p.cwiseProduct(v).cwiseProduct(p);
    }
};

#endif