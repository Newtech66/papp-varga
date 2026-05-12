#ifndef CONE_PRODUCT_CONES_H
#define CONE_PRODUCT_CONES_H
#include <memory>
#include <Eigen/Core>
#include "cone_parameters.hpp"
#include "common_typedefs.hpp"
#include "cone_dispatch.hpp"
#include <vector>
#include <string>

template<typename prec_type>
class ConeProduct{
protected:
    std::vector<std::pair<std::string, std::unique_ptr<ConeParameters>>> cones;
    bool is_symmetric = true;
public:
    ConeProduct() = default;
    // https://stackoverflow.com/questions/8164567/how-to-make-my-custom-type-to-work-with-range-based-for-loops
    auto begin(){return cones.begin();}
    auto end(){return cones.end();}
    auto begin(){return cones.cbegin();}
    auto end(){return cones.cend();}
    auto begin() const {return cones.begin();}
    auto end() const {return cones.end();}
    void addCone(std::string cone_id, std::unique_ptr<ConeParameters> cone_params){
        cones.emplace_back(cone_id, std::move(cone_params));
        is_symmetric &= cone_is_symmetric(cone_id);
    }
    std::string coneName() const override{
        std::string name("Product of the following cones:\n");
        for(auto& [cone, cone_params]: cones){
            name += cone->coneName() + " with " + std::string(cone_params->numVariables()) + " variables\n";
        }
        return name;
    }
    template<typename Derived>
    optVector<prec_type> grad(const Eigen::MatrixBase<Derived>& p){
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        for(auto& [cone_id, cone_params]: cones){
            int nvar = cone_params->numVariables();
            out(Eigen::seqN(cpos, nvar)) = cone_grad(cone_id, cone_params, p(Eigen::seqN(cpos, nvar)));
            cpos += nvar;
        }
        return out;
    }
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&){
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        for(auto& [cone_id, cone_params]: cones){
            int nvar = cone_params->numVariables();
            out(Eigen::seqN(cpos, nvar)) = cone_hvp(cone_id, cone_params, p(Eigen::seqN(cpos, nvar)));
            cpos += nvar;
        }
        return out;
    }
    template<typename Derived>
    optVector<prec_type> ihvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&){
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        for(auto& [cone_id, cone_params]: cones){
            int nvar = cone_params->numVariables();
            out(Eigen::seqN(cpos, nvar)) = cone_ihvp(cone_id, cone_params, p(Eigen::seqN(cpos, nvar)));
            cpos += nvar;
        }
        return out;
    }
};

#endif