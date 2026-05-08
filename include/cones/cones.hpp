#ifndef CONES_PAPP_VARGA_H
#define CONES_PAPP_VARGA_H
#include <Eigen/Core>
#include "common_typedefs.hpp"
#include <memory>

template<typename RealScalar>
class Cone{
protected:
    int barrier_parameter;
    int num_params;
public:
    int barrierParameter(){return barrier_parameter;}
    int numParams(){return num_params;}
    virtual std::string coneName() = 0;
    // this returns the current point
    virtual optVector<RealScalar> point() const = 0;
    // this updates the current point
    virtual void updatePoint(const Eigen::Ref<const optVector<RealScalar>>&) = 0;
    // this returns the gradient evaluated at the current point
    virtual optVector<RealScalar> jacobian() = 0;
    // this returns the hessian-vector product with v evaluated at the current point
    virtual optVector<RealScalar> hvp(const Eigen::Ref<const optVector<RealScalar>>&) = 0;
    virtual optVector<RealScalar> ihvp(const Eigen::Ref<const optVector<RealScalar>>&){
        const std::string error_message = "Inverse Hessian-vector product is not implemented for " + coneName();
        throw std::logic_error(error_message.data());
    }
};

template<typename RealScalar>
using cone_array = std::vector<std::unique_ptr<Cone<RealScalar>>>;

template<typename RealScalar>
class ConeProduct : public Cone<RealScalar>{
protected:
    cone_array<RealScalar> cones;
    optVector<RealScalar> p, jac;
public:
    ConeProduct(){}
    ConeProduct(cone_array<RealScalar>& cones);
    optVector<RealScalar> point() const override{return p;}
    void updatePoint(const Eigen::Ref<const optVector<RealScalar>>& v) override;
    optVector<RealScalar> jacobian() override{return jac;}
    optVector<RealScalar> hvp(const Eigen::Ref<const optVector<RealScalar>>& v) override;
    optVector<RealScalar> ihvp(const Eigen::Ref<const optVector<RealScalar>>& v) override;
};

#endif