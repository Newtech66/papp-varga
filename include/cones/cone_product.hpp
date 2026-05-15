#ifndef CONE_PRODUCT_CONES_H
#define CONE_PRODUCT_CONES_H
#include <Eigen/Core>
#include "cone.hpp"
#include "psd_parameters.hpp"
#include "common_typedefs.hpp"
#include <vector>
#include <string>
#include <array>

enum class ConeTypes {
    REALPSD = 0,
    COMPLEXPSD = 1,
};

template<typename prec_type>
class ConeProduct{
private:
    static constexpr int cone_type_count = 2;
    std::vector<PSDParameters> real_psd, complex_psd;
protected:
    std::vector<ConeTypes> cones;
    int barrier_parameter = 0;
public:
    ConeProduct() = default;
    // concept here?
    template<ConeTypes cone_id>
    void addCone(auto cone_params){
        using enum ConeTypes;
        cones.push_back(cone_id);
        if constexpr (cone_id == REALPSD)   real_psd.push_back(cone_params);
        else if constexpr(cone_id == COMPLEXPSD) complex_psd.push_back(cone_params);
        barrier_parameter += cone_params.barrierParameter();
    }
    int barrierParameter() const{return barrier_parameter;}
    // probably should use a concept here to prevent a bad f being passed
    template<typename transform_func>
    void func_over_cones(transform_func f){
        using enum ConeTypes;
        std::array<int, cone_type_count> counter{0};
        for(const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    f.template operator()<REALPSD, PSD<prec_type, false>>(real_psd[counter[static_cast<int>(cone_id)]]);
                    break;
                case COMPLEXPSD:
                    f.template operator()<COMPLEXPSD, PSD<prec_type, true>>(complex_psd[counter[static_cast<int>(cone_id)]]);
                    break;
            }
            counter[static_cast<int>(cone_id)]++;
        }
    }
    // only symmetric cones should be listed here
    template<typename transform_func>
    void func_over_symmetric_cones(transform_func f){
        using enum ConeTypes;
        std::array<int, cone_type_count> counter{0};
        for(const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    f.template operator()<REALPSD, PSD<prec_type, false>>(real_psd[counter[static_cast<int>(cone_id)]]);
                    break;
                case COMPLEXPSD:
                    f.template operator()<COMPLEXPSD, PSD<prec_type, true>>(complex_psd[counter[static_cast<int>(cone_id)]]);
                    break;
            }
            counter[static_cast<int>(cone_id)]++;
        }
    }
    std::string coneName(){
        using enum ConeTypes;
        std::string name("Product of the following cones:\n");
        auto add_name = [&name]<ConeTypes cone_id, Cone U>(auto& cone_params){
            name += cone_params.coneName() + "\n";
        };
        func_over_cones(add_name);
        return name;
    }
    optVector<prec_type> grad(const optVector<prec_type>& p){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        auto cone_grad = [&p, &out, &cpos]<ConeTypes cone_id, Cone U>(auto& cone_params){
            int nvar = cone_params.numVariables();
            out(Eigen::seqN(cpos, nvar)) = U::grad(p, cone_params);
            cpos += nvar;
        };
        func_over_cones(cone_grad);
        return out;
    }
    optVector<prec_type> hvp(const optVector<prec_type>& p, const optMatrix<prec_type>& vecs){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        auto cone_hvp = [&p, &out, &cpos]<ConeTypes cone_id, Cone U>(auto& cone_params){
            int nvar = cone_params.numVariables();
            out(Eigen::seqN(cpos, nvar)) = U::hvp(p, vecs, cone_params);
            cpos += nvar;
        };
        func_over_cones(cone_hvp);
        return out;
    }
    optVector<prec_type> ihvp(const optVector<prec_type>& p, const optMatrix<prec_type>& vecs){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        auto cone_ihvp = [&p, &out, &cpos]<ConeTypes cone_id, Cone U>(auto& cone_params){
            int nvar = cone_params.numVariables();
            out(Eigen::seqN(cpos, nvar)) = U::ihvp(p, vecs, cone_params);
            cpos += nvar;
        };
        func_over_cones(cone_ihvp);
        return out;
    }
    // these methods are only implemented by symmetric cones
    void get_nt_scaling(const optVector<prec_type>& s, const optVector<prec_type>& z,
        std::vector<optMatrix<prec_type>>& scaling_matrix, optVector<prec_type>& scaling_point, optVector<prec_type>& scaled_variable){
        using enum ConeTypes;
        scaling_point.resize(s.rows());
        scaled_variable.resize(s.rows());
        int cpos = 0;
        int i = 0;
        auto sym_cone_get_nt_scaling = [&s, &z, &scaling_matrix, &scaling_point, &scaled_variable, &cpos, &i]<ConeTypes cone_id, Cone U>(auto& cone_params){
            int nvar = cone_params.numVariables();
            idxs = Eigen::seqN(cpos, nvar);
            scaling_matrix[i].resize(nvar, nvar);
            U::get_nt_scaling(s, z, scaling_matrix[i], scaling_point(idxs), scaled_variable(idxs), cone_params);
            cpos += nvar;
            i++;
        };
        func_over_symmetric_cones(sym_cone_get_nt_scaling);
    }
    prec_type get_nt_step_length(const optVector<prec_type>& s, const optVector<prec_type>& z, const optVector<prec_type>& scaled_variable){
        using enum ConeTypes;
        std::vector<prec_type> alphas;
        int cpos = 0;
        auto sym_cone_get_nt_step_length = [&s, &z, &scaled_variable, &cpos]<ConeTypes cone_id, Cone U>(auto& cone_params){
            int nvar = cone_params.numVariables();
            idxs = Eigen::seqN(cpos, nvar);
            alphas.push_back(U::get_nt_step_length(s, z, scaled_variable(idxs), cone_params));
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_get_nt_step_length);
        return *std::min_element(alphas.begin(), alphas.end());
    }
};

#endif