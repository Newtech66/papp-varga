#include "cone_product.hpp"

template<typename prec_type>
ConeProduct<prec_type>::ConeProduct(cone_array<prec_type>& cones){
    std::swap(this->cones, cones);
    this->barrier_parameter = 0;
    this->num_params = 0;
    for(auto&& cone : this->cones){
        this->barrier_parameter += cone->barrierParameter();
        this->num_params += cone->numParams();
    }
    // let's initialize the point and jacobian
    p = optVector<prec_type>::Zero(this->num_params);
    jac = optVector<prec_type>::Zero(this->num_params);
    hvpsto = optVector<prec_type>::Zero(this->num_params);
    int cpos = 0;
    for(auto&& cone : this->cones){
        p(Eigen::seqN(cpos, cone->numParams())) = cone->point();
        jac(Eigen::seqN(cpos, cone->numParams())) = cone->jacobian();
        cpos += cone->numParams();
    }
    jac_updated = true;
}

template <typename prec_type>
void ConeProduct<prec_type>::updatePoint(const Eigen::Ref<const optVector<prec_type>> &v){
    int cpos = 0;
    for(auto&& cone : cones){
        cone->updatePoint(v(Eigen::seqN(cpos, cone->numParams())));
        p(Eigen::seqN(cpos, cone->numParams())) = cone->point();
        cpos += cone->numParams();
    }
    jac_updated = false;
}

template <typename prec_type>
optVector<prec_type> ConeProduct<prec_type>::jacobian(){
    if(!jac_updated){
        int cpos = 0;
        for(auto&& cone : cones){
            jac(Eigen::seqN(cpos, cone->numParams())) = cone->jacobian();
            cpos += cone->numParams();
        }
        jac_updated = true;
    }
    return jac;
}

template<typename prec_type>
optVector<prec_type> ConeProduct<prec_type>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
        // perform the hessian-vector product for each segment
        int cpos = 0;
        for(auto&& cone : cones){
            hvpsto(Eigen::seqN(cpos, cone->numParams())) = cone->hvp(v(Eigen::seqN(cpos, cone->numParams())));
            cpos += cone->numParams();
        }
        return hvpsto;
}

template<typename prec_type>
optVector<prec_type> ConeProduct<prec_type>::ihvp(const Eigen::Ref<const optVector<prec_type>>& v){
        // perform the hessian-vector product for each segment
        optVector<prec_type> ihvp = optVector<prec_type>::Zero(this->num_params);
        int cpos = 0;
        for(auto&& cone : cones){
            ihvp(Eigen::seqN(cpos, cone->numParams())) = cone->ihvp(v(Eigen::seqN(cpos, cone->numParams())));
            cpos += cone->numParams();
        }
        return ihvp;
}