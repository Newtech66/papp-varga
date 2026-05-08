#ifndef POSSEMIDEFINITE_PAPP_VARGA_H
#define POSSEMIDEFINITE_PAPP_VARGA_H
#include "cones.hpp"
#include "common_typedefs.hpp"
#include <type_traits>

template<typename prec_type, bool is_complex>
class PositiveSemidefinite : public Cone<prec_type>{
protected:
    int matrix_size;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> P, Pinv, iden;
    bool jac_updated = false;
public:
    PositiveSemidefinite(const int n);
    std::string coneName() const override{return (is_complex ? std::string("Complex positive semi-definite cone") : std::string("Real positive semi-definite cone"));}
    optVector<prec_type> point() const override;
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override;
    optVector<prec_type> jacobian() override;
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override;
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override;
};

template<typename prec_type>
class DiagonalPositiveSemidefinite : public Cone<prec_type>{
protected:
    int matrix_size;
    optVector<prec_type> p, pinv;
public:
    DiagonalPositiveSemidefinite(const int n);
    std::string coneName() const override{return std::string("Diagonal real positive semi-definite cone");}
    optVector<prec_type> point() const override{return p;}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override;
    optVector<prec_type> jacobian() override{return -pinv;}
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return pinv.cwiseProduct(v).cwiseProduct(pinv);
    }
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override{
        return p.cwiseProduct(v).cwiseProduct(p);
    }
};

#endif