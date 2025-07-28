#ifndef CONES_PAPP_VARGA_H
#define CONES_PAPP_VARGA_H
#include <Eigen/Dense>
#include "misc_exceptions.cpp"

template<typename RealScalar>
class Cone{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    int barrier_parameter;
    int num_params;
public:
    int barrierParameter(){return barrier_parameter;}
    int numParams(){return num_params;}
    // this returns the current point
    virtual Vector point() const = 0;
    // this updates the current point
    virtual void updatePoint(const Eigen::Ref<const Vector>&) = 0;
    // this returns the gradient evaluated at the current point
    virtual Vector jacobian() const = 0;
    // this returns the hessian-vector product with v evaluated at the current point
    virtual Vector hvp(const Eigen::Ref<const Vector>&) const = 0;
    virtual Vector ihvp(const Eigen::Ref<const Vector>&) const{throw NotImplemented();}
};

template<typename RealScalar>
class ConeProduct : public Cone<RealScalar>{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    std::vector<std::unique_ptr<Cone<RealScalar>>> cones;
    Vector p, jac;
public:
    ConeProduct(){}
    ConeProduct(std::vector<std::unique_ptr<Cone<RealScalar>>>& cones){
        this->barrier_parameter = 0;
        this->num_params = 0;
        // a bit unsure about the syntax here, how do you go over a container of unique_ptr's?
        for(auto&& cone : cones){
            this->barrier_parameter += cone->barrierParameter();
            this->num_params += cone->numParams();
        }
        // let's initialize the point and jacobian
        p = Vector::Zero(this->num_params);
        jac = Vector::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            p.segment(cpos, cone->numParams()) = cone->point();
            jac.segment(cpos, cone->numParams()) = cone->jacobian();
            cpos += cone->numParams();
        }
        for(auto&& cone : cones){
            this->cones.push_back(std::move(cone));
        }
    }
    Vector point() const override{return p;}
    void updatePoint(const Eigen::Ref<const Vector>& v) override{
        int cpos = 0;
        for(auto&& cone : cones){
            cone->updatePoint(v.segment(cpos, cone->numParams()));
            p.segment(cpos, cone->numParams()) = cone->point();
            jac.segment(cpos, cone->numParams()) = cone->jacobian();
            cpos += cone->numParams();
        }    
    }
    Vector jacobian() const override{return jac;}
    Vector hvp(const Eigen::Ref<const Vector>& v) const override{
        // perform the hessian-vector product for each segment
        Vector hvp = Vector::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            hvp.segment(cpos, cone->numParams()) = cone->hvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return hvp;
    }
    Vector ihvp(const Eigen::Ref<const Vector>& v) const override{
        // perform the hessian-vector product for each segment
        Vector ihvp = Vector::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            ihvp.segment(cpos, cone->numParams()) = cone->ihvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return ihvp;
    }
};

#endif