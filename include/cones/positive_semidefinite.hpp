#ifndef POSITIVE_SEMIDEFINITE_CONES_H
#define POSITIVE_SEMIDEFINITE_CONES_H
#include "cone.hpp"
#include "common_typedefs.hpp"

template<typename prec_type, bool complex_type>
class PositiveSemidefinite : public Cone{
protected:
    int matrix_size;
    bool is_symmetric = true;
    bool is_complex = complex_type;
public:
    PositiveSemidefinite(const int n);
    static std::string coneId() const override{return "PSD";}
    static std::string coneName() const override{
        return (is_complex ? std::string("Complex positive semi-definite cone") :
        std::string("Real positive semi-definite cone"));
    }
    template<typename Derived>
    static optVector<prec_type> grad(const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    static optVector<prec_type> hvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
    template<typename Derived>
    static optVector<prec_type> ihvp(const Eigen::MatrixBase<Derived>&, const Eigen::MatrixBase<Derived>&) override;
};

#endif