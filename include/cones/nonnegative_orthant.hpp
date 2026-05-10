#ifndef NONNEGATIVE_ORTHANT_CONES_H
#define NONNEGATIVE_ORTHANT_CONES_H
#include "cone.hpp"
#include "common_typedefs.hpp"

template<typename prec_type>
class NonnegativeOrthant : public Cone<prec_type>{
protected:
    int matrix_size;
    bool is_symmetric = true;
    bool is_complex = false;
public:
    NonnegativeOrthant(const int n);
    std::string coneName() const override{return std::string("Non-negative orthant cone");}
    template<typename Derived>
    optVector<prec_type> grad(const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    optVector<prec_type> ihvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
};

#endif