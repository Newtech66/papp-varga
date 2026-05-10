#ifndef CONE_PRODUCT_CONES_H
#define CONE_PRODUCT_CONES_H
#include <memory>
#include <Eigen/Core>
#include "cone.hpp"
#include "common_typedefs.hpp"

using conearray = std::vector<std::unique_ptr<Cone>>;

template<typename prec_type>
class ConeProduct : public Cone{
protected:
    conearray cones;
public:
    ConeProduct(){}
    ConeProduct(cone_array& cones);
    std::string coneName() const override;
    template<typename Derived>
    optVector<prec_type> grad(const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    optVector<prec_type> ihvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
};

template<typename prec_type>
std::string ConeProduct<prec_type>::coneName() const{
    std::string name("Product of the following cones:\n");
    for(auto&& cone: cones){
        name += cone->coneName() + " with " + std::string(cone->numParams()) + " parameters\n";
    }
    return name;
}

#endif