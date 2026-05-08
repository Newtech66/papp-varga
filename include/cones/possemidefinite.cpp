#ifndef POSSEMIDEFINITE_PAPP_VARGA_H
#define POSSEMIDEFINITE_PAPP_VARGA_H
#include "vectorize.cpp"
#include "cones.hpp"
#include "common_typedefs.hpp"
#include "complex_deduction.hpp"

template<typename Scalar>
class PositiveSemidefinite : public Cone<Scalar>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return type is RealVector
    using prec_type = typename (is_complex_t<Scalar>() ? Scalar::value_type : Scalar);
    using RealVector = optVector<prec_type>;
protected:
    int matrix_size;
    optMatrix<Scalar> P, Pinv, I;
    bool jac_updated = false;
public:
    PositiveSemidefinite(const int n) : matrix_size(n){
        I.setIdentity(matrix_size, matrix_size);
        P.setIdentity(matrix_size, matrix_size);
        this->barrier_parameter = matrix_size;
        if(is_complex_t<Scalar>())  this->num_params = 2 * matrix_size * matrix_size;
        else    this->num_params = matrix_size * matrix_size;
    }
    std::string coneName() override{
        if(is_complex_t<Scalar>())  return "Complex positive semi-definite cone";
        return "Real positive semi-definite cone";
    }
    RealVector point() const override{return vec<RealScalar>(P);}
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        if(is_complex_t<Scalar>())  P = unvecComplex(p, matrix_size);
        else    P = unvecReal(p, matrix_size);
        jac_updated = false;
    }
    RealVector jacobian() override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(I);
            jac_updated = true;
        }
        return -vec<base_type>(Pinv);
    }
    RealVector hvp(const Eigen::Ref<const RealVector>& v) override{
        if(!jac_updated){
            Pinv = P.ldlt().solve(I);
            jac_updated = true;
        }
        if(is_complex_t<Scalar>())  return vec<base_type>(Pinv * unvecComplex(v, matrix_size) * Pinv);
        else    return vec<base_type>(Pinv * unvecReal(v, matrix_size) * Pinv);
    }
    RealVector ihvp(const Eigen::Ref<const RealVector>& v) override{
        if(is_complex_t<Scalar>())  return vec<base_type>(P * unvecComplex(v, matrix_size) * P);
        else    return vec<base_type>(P * unvecReal(v, matrix_size) * P);
    }
};

template<typename RealScalar>
class DiagonalPositiveSemidefinite : public Cone<RealScalar>{
protected:
    int matrix_size;
    optVector<RealScalar> p, pinv;
public:
    DiagonalPositiveSemidefinite(const int n) : matrix_size(n){
        p.setOnes(matrix_size);
        pinv.setOnes(matrix_size);
        this->barrier_parameter = matrix_size;
        this->num_params = matrix_size;
    }
    std::string coneName() override{return "Diagonal real positive semi-definite cone";}
    optVector<RealScalar> point() const override{return p;}
    void updatePoint(const Eigen::Ref<const optVector<RealScalar>>& p) override{
        this->p = p;
        pinv = p.inverse();
    }
    optVector<RealScalar> jacobian() override{return -pinv;}
    optVector<RealScalar> hvp(const Eigen::Ref<const Vector>& v) override{
        return pinv.cwiseProduct(v).cwiseProduct(pinv);
    }
    optVector<RealScalar> ihvp(const Eigen::Ref<const Vector>& v) override{
        return p.cwiseProduct(v).cwiseProduct(p);
    }
};

#endif