#include "cones.hpp"

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

template class Cone<double>;
template class ConeProduct<double>;