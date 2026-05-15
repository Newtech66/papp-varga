#ifndef CONE_PRODUCT_CONES_H
#define CONE_PRODUCT_CONES_H
#include <Eigen/Core>
#include "cone_parameters.hpp"
#include "common_typedefs.hpp"
#include <vector>
#include <string>
#include <format>
#include <array>
#include <map>

enum class ConeTypes {
    REALPSD = 0,
    COMPLEXPSD = 1,
    REALLPE = 2,
    COMPLEXLPE = 3,
    NONNEGORTH = 4
};

const std::map<ConeTypes, std::string> cone_ids{
    {ConeTypes::REALPSD, "REALPSD"},
    {ConeTypes::COMPLEXPSD, "COMPLEXPSD"},
    {ConeTypes::REALLPE, "REALLPE"},
    {ConeTypes::COMPLEXLPE, "COMPLEXLPE"},
    {ConeTypes::NONNEGORTH, "NONNEGORTH"}
};

template<typename prec_type>
class ConeProduct{
private:
    static constexpr int cone_type_count = 5;
    std::vector<RealPSDParameters> real_psd;
    std::vector<ComplexPSDParameters> complex_psd;
    std::vector<RealLPEParameters> real_lpe;
    std::vector<ComplexLPEParameters> complex_lpe;
    std::vector<NonNegOrthParameters> nonneg_orth;
protected:
    std::vector<ConeTypes> cones;
    bool is_symmetric = true;
public:
    ConeProduct() = default;
    template<typename cone_type>
    void addCone(ConeTypes cone_id, cone_type cone_params){
        using enum ConeTypes;
        cones.push_back(cone_id);
        switch(cone_id){
            case REALPSD:
                real_psd.push_back(cone_params);
                break;
            case COMPLEXPSD:
                complex_psd.push_back(cone_params);
                break;
            case REALLPE:
                real_lpe.push_back(cone_params);
                break;
            case COMPLEXLPE:
                complex_lpe.push_back(cone_params);
                break;
            case NONNEGORTH:
                nonneg_orth.push_back(cone_params);
                break;
        }
    }
    std::string coneName() const{
        using enum ConeTypes;
        std::string name("Product of the following cones:\n");
        std::array<int, cone_type_count> counter{0};
        for(const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    name += real_psd[counter[cone_id]].coneName() + "\n";
                    break;
                case COMPLEXPSD:
                    name += complex_psd[counter[cone_id]].coneName() + "\n";
                    break;
                case REALLPE:
                    name += real_lpe[counter[cone_id]].coneName() + "\n";
                    break;
                case COMPLEXLPE:
                    name += complex_lpe[counter[cone_id]].coneName() + "\n";
                    break;
                case NONNEGORTH:
                    name += nonneg_orth[counter[cone_id]].coneName() + "\n";
                    break;
            }
            counter[cone_id]++;
        }
    }
    template<typename Derived>
    optVector<prec_type> grad(const Eigen::MatrixBase<Derived>& p){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        std::array<int, cone_type_count> counter{0};
        for(int cpos = 0, nvar;const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    auto& cone_params = real_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, false>::grad(p, cone_params);
                    break;
                case COMPLEXPSD:
                    auto& cone_params = complex_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, true>::grad(p, cone_params);
                    break;
                case REALLPE:
                    auto& cone_params = real_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, false>::grad(p, cone_params);
                    break;
                case COMPLEXLPE:
                    auto& cone_params = complex_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, true>::grad(p, cone_params);
                    break;
                case NONNEGORTH:
                    auto& cone_params = nonneg_orth[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = NonNegOrth<prec_type>::grad(p, cone_params);
                    break;
            }
            cpos += nvar;
            counter[cone_id]++;
        }
        return out;
    }
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>& p, const Eigen::MatrixBase<Derived>& vecs){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        std::array<int, cone_type_count> counter{0};
        for(int cpos = 0, nvar;const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    auto& cone_params = real_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, false>::hvp(p, vecs, cone_params);
                    break;
                case COMPLEXPSD:
                    auto& cone_params = complex_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, true>::hvp(p, vecs, cone_params);
                    break;
                case REALLPE:
                    auto& cone_params = real_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, false>::hvp(p, vecs, cone_params);
                    break;
                case COMPLEXLPE:
                    auto& cone_params = complex_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, true>::hvp(p, vecs, cone_params);
                    break;
                case NONNEGORTH:
                    auto& cone_params = nonneg_orth[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = NonNegOrth<prec_type>::hvp(p, vecs, cone_params);
                    break;
            }
            cpos += nvar;
            counter[cone_id]++;
        }
        return out;
    }
    template<typename Derived>
    optVector<prec_type> ihvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&){
        using enum ConeTypes;
        optVector<prec_type> out;
        out.resize(p.size());
        std::array<int, cone_type_count> counter{0};
        for(int cpos = 0, nvar;const auto& cone_id: cones){
            switch(cone_id){
                case REALPSD:
                    auto& cone_params = real_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, false>::ihvp(p, vecs, cone_params);
                    break;
                case COMPLEXPSD:
                    auto& cone_params = complex_psd[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = PSD<prec_type, true>::ihvp(p, vecs, cone_params);
                    break;
                case REALLPE:
                    auto& cone_params = real_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, false>::ihvp(p, vecs, cone_params);
                    break;
                case COMPLEXLPE:
                    auto& cone_params = complex_lpe[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = LPE<prec_type, true>::ihvp(p, vecs, cone_params);
                    break;
                case NONNEGORTH:
                    auto& cone_params = nonneg_orth[counter[cone_id]];
                    nvar = cone_params.numVariables();
                    out(Eigen::seqN(cpos, nvar)) = NonNegOrth<prec_type>::ihvp(p, vecs, cone_params);
                    break;
            }
            cpos += nvar;
            counter[cone_id]++;
        }
        return out;
    }
};

#endif