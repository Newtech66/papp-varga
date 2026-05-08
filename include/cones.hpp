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
    optVector<prec_type> p, jac;
public:
    ConeProduct(){}
    ConeProduct(cone_array<prec_type>& cones);
    std::string coneName() const override;
    optVector<prec_type> point() const override {return p;}
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& v) override;
    optVector<prec_type> jacobian() override{return jac;}
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

template<typename prec_type>
ConeProduct<prec_type>::ConeProduct(cone_array<prec_type>& cones){
    this->cones.insert(this->cones.end(), std::make_move_iterator(cones.begin()), std::make_move_iterator(cones.end()));
    this->barrier_parameter = 0;
    this->num_params = 0;
    for(auto&& cone : this->cones){
        this->barrier_parameter += cone->barrierParameter();
        this->num_params += cone->numParams();
    }
    // let's initialize the point and jacobian
    p = optVector<prec_type>::Zero(this->num_params);
    jac = optVector<prec_type>::Zero(this->num_params);
    int cpos = 0;
    for(auto&& cone : this->cones){
        p.segment(cpos, cone->numParams()) = cone->point();
        jac.segment(cpos, cone->numParams()) = cone->jacobian();
        cpos += cone->numParams();
    }
}

template <typename prec_type>
void ConeProduct<prec_type>::updatePoint(const Eigen::Ref<const optVector<prec_type>> &v)
{
    int cpos = 0;
    for(auto&& cone : cones){
        cone->updatePoint(v.segment(cpos, cone->numParams()));
        p.segment(cpos, cone->numParams()) = cone->point();
        jac.segment(cpos, cone->numParams()) = cone->jacobian();
        cpos += cone->numParams();
    }
}

template<typename prec_type>
optVector<prec_type> ConeProduct<prec_type>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
        // perform the hessian-vector product for each segment
        optVector<prec_type> hvp = optVector<prec_type>::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            hvp.segment(cpos, cone->numParams()) = cone->hvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return hvp;
}

template<typename prec_type>
optVector<prec_type> ConeProduct<prec_type>::ihvp(const Eigen::Ref<const optVector<prec_type>>& v){
        // perform the hessian-vector product for each segment
        optVector<prec_type> ihvp = optVector<prec_type>::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            ihvp.segment(cpos, cone->numParams()) = cone->ihvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return ihvp;
}

#endif