#ifndef CONES_PAPP_VARGA_H
#define CONES_PAPP_VARGA_H
#include <memory>
#include <Eigen/Core>
#include "common_typedefs.hpp"

template<typename prec_type>
class Cone{
protected:
    int barrier_parameter;
    int num_params;
public:
    int barrierParameter() const {return this->barrier_parameter;}
    int numParams() const {return this->num_params;}
    virtual std::string coneName() const = 0;
    // this returns the current point
    virtual optVector<prec_type> point() const = 0;
    // this updates the current point
    virtual void updatePoint(const Eigen::Ref<const optVector<prec_type>>&) = 0;
    // this returns the gradient evaluated at the current point
    virtual optVector<prec_type> jacobian() = 0;
    // this returns the hessian-vector product with v evaluated at the current point
    virtual optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>&) = 0;
    virtual optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>&){
        const std::string error_message = "Inverse Hessian-vector product is not implemented for " + coneName();
        throw std::logic_error(error_message.data());
    }
};

template<typename prec_type>
using cone_array = std::vector<std::unique_ptr<Cone<prec_type>>>;

template<typename prec_type>
class ConeProduct : public Cone<prec_type>{
protected:
    cone_array<prec_type> cones;
    bool jac_updated;
    optVector<prec_type> p, jac, hvpsto;
public:
    ConeProduct(){}
    ConeProduct(cone_array<prec_type>& cones);
    std::string coneName() const override;
    optVector<prec_type> point() const override {return p;}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& v) override;
    optVector<prec_type> jacobian() override;
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override;
    optVector<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& v) override;
};

template<typename prec_type>
std::string ConeProduct<prec_type>::coneName() const{
    std::string name("Product of the following cones:\n");
    for(auto&& cone: cones){
        name += cone->coneName() + "\n";
    }
    return name;
}

#endif