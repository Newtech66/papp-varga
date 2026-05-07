#include "cones.hpp"
#include "common_typedefs.hpp"

template<typename RealScalar>
ConeProduct<RealScalar>::ConeProduct(cone_array<RealScalar>& cones){
    this->cones.insert(this->cones.end(), std::make_move_iterator(cones.begin()), std::make_move_iterator(cones.end()))
    this->barrier_parameter = 0;
    this->num_params = 0;
    for(auto&& cone : this->cones){
        this->barrier_parameter += cone->barrierParameter();
        this->num_params += cone->numParams();
    }
    // let's initialize the point and jacobian
    p = Vector::Zero(this->num_params);
    jac = Vector::Zero(this->num_params);
    for(int cpos = 0; auto&& cone : this->cones){
        p.segment(cpos, cone->numParams()) = cone->point();
        jac.segment(cpos, cone->numParams()) = cone->jacobian();
        cpos += cone->numParams();
    }
}

template <typename RealScalar>
void ConeProduct<RealScalar>::updatePoint(const Eigen::Ref<const optVector<RealScalar>> &v)
{
    for(int cpos = 0; auto&& cone : cones){
        cone->updatePoint(v.segment(cpos, cone->numParams()));
        p.segment(cpos, cone->numParams()) = cone->point();
        jac.segment(cpos, cone->numParams()) = cone->jacobian();
        cpos += cone->numParams();
    }
}

template<typename RealScalar>
optVector<RealScalar> ConeProduct<RealScalar>::hvp(const Eigen::Ref<const optVector<RealScalar>>& v){
        // perform the hessian-vector product for each segment
        Vector hvp = Vector::Zero(this->num_params);
        for(int cpos = 0; auto&& cone : cones){
            hvp.segment(cpos, cone->numParams()) = cone->hvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return hvp;
}

template<typename RealScalar>
optVector<RealScalar> ConeProduct<RealScalar>::ihvp(const Eigen::Ref<const optVector<RealScalar>>& v){
        // perform the hessian-vector product for each segment
        Vector ihvp = Vector::Zero(this->num_params);
        for(int cpos = 0; auto&& cone : cones){
            ihvp.segment(cpos, cone->numParams()) = cone->ihvp(v.segment(cpos, cone->numParams()));
            cpos += cone->numParams();
        }
        return ihvp;
}