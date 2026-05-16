#ifndef CONE_PRODUCT_CONES_H
#define CONE_PRODUCT_CONES_H
#include <Eigen/Core>
#include "common_typedefs.hpp"
#include <vector>
#include <string>
#include <array>

enum class ConeTypes {
    REALPSD = 0,
    COMPLEXPSD = 1,
};

template<typename prec_type, bool is_complex> class PSD;

template<typename prec_type>
class ConeProduct{
private:
    static constexpr int cone_type_count = 2;
    std::vector<PSD<prec_type, false>> real_psd;
    std::vector<PSD<prec_type, true>> complex_psd;
protected:
    std::vector<ConeTypes> cones;
    int barrier_parameter = 0;

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
public:
    ConeProduct() = default;
    // concept here?
    // this is not very safe, a user can accidentally pass REALPSD but PSD<prec_type, true>
    template<ConeTypes cone_id>
    void addCone(auto cone){
        using enum ConeTypes;
        cones.push_back(cone_id);
        if constexpr (cone_id == REALPSD)   real_psd.push_back(cone);
        else if constexpr(cone_id == COMPLEXPSD) complex_psd.push_back(cone);
        barrier_parameter += cone.barrierParameter();
    }
    int barrierParameter() const{return barrier_parameter;}
    std::string coneName(){
        using enum ConeTypes;
        std::string name("Product of the following cones:\n");
        auto add_name = [&name]<ConeTypes cone_id, typename U>(U& cone){
            name += cone.coneName() + "\n";
        };
        func_over_cones(add_name);
        return name;
    }
    optVector<prec_type> grad(const Eigen::Ref<const optVector<prec_type>>& p){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        int cpos = 0;
        auto cone_grad = [&p, &out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            out(Eigen::seqN(cpos, nvar)) = cone.grad(p);
            cpos += nvar;
        };
        func_over_cones(cone_grad);
        return out;
    }
    optMatrix<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& vecs){
        using enum ConeTypes;
        optMatrix<prec_type> out;
        out.resize(p.rows(), vecs.cols());
        int cpos = 0;
        auto cone_hvp = [&p, &vecs, &out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            out(Eigen::seqN(cpos, nvar), Eigen::placeholders::all) = cone.hvp(p, vecs);
            cpos += nvar;
        };
        func_over_cones(cone_hvp);
        return out;
    }
    optMatrix<prec_type> ihvp(const Eigen::Ref<const optVector<prec_type>>& p, const Eigen::Ref<const optMatrix<prec_type>>& vecs){
        using enum ConeTypes;
        optMatrix<prec_type> out;
        out.resize(p.rows(), vecs.cols());
        int cpos = 0;
        auto cone_ihvp = [&p, &vecs, &out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            out(Eigen::seqN(cpos, nvar), Eigen::placeholders::all) = cone.ihvp(p, vecs);
            cpos += nvar;
        };
        func_over_cones(cone_ihvp);
        return out;
    }
    // these methods are only implemented by symmetric cones
    void update_nt_scaling(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z){
        using enum ConeTypes;
        int cpos = 0;
        auto sym_cone_update_nt_scaling = [&s, &z, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            auto idxs = Eigen::seqN(cpos, nvar);
            cone.update_nt_scaling(s(idxs), z(idxs));
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_update_nt_scaling);
    }
    prec_type get_nt_step_length(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z){
        using enum ConeTypes;
        std::vector<prec_type> alphas;
        int cpos = 0;
        auto sym_cone_get_nt_step_length = [&s, &z, &cpos, &alphas]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            auto idxs = Eigen::seqN(cpos, nvar);
            alphas.push_back(cone.get_nt_step_length(s(idxs), z(idxs)));
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_get_nt_step_length);
        return *std::min_element(alphas.begin(), alphas.end());
    }
    optVector<prec_type> get_nt_rhs_s(const Eigen::Ref<const optVector<prec_type>>& s, const Eigen::Ref<const optVector<prec_type>>& z, const prec_type centering_parameter, const prec_type mu){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(s.size());
        int cpos = 0;
        auto sym_cone_get_nt_rhs_s = [&s, &z, &centering_parameter, &mu, &out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            auto idxs = Eigen::seqN(cpos, nvar);
            out(idxs) = cone.get_nt_rhs_s(s, z, centering_parameter, mu);
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_get_nt_rhs_s);
        return out;
    }
    optVector<prec_type> get_nt_scaling_point(const int ssize){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(ssize);
        int cpos = 0;
        auto sym_cone_get_nt_scaling_point = [&out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            auto idxs = Eigen::seqN(cpos, nvar);
            out(idxs) = cone.get_nt_scaling_point();
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_get_nt_scaling_point);
        return out;
    }
    optVector<prec_type> get_nt_scaled_variable(const int ssize){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(ssize);
        int cpos = 0;
        auto sym_cone_get_nt_scaled_variable = [&out, &cpos]<ConeTypes cone_id, typename U>(U& cone){
            int nvar = cone.numVariables();
            auto idxs = Eigen::seqN(cpos, nvar);
            out(idxs) = cone.get_nt_scaled_variable();
            cpos += nvar;
        };
        func_over_symmetric_cones(sym_cone_get_nt_scaled_variable);
        return out;
    }
};

#endif