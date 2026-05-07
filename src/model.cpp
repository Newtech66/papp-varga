#ifndef MODEL_PAPP_VARGA_H
#define MODEL_PAPP_VARGA_H
#include "cones.hpp"
#include "common_typedefs.hpp"

template<typename RealScalar>
class Model{
private:
    ConeProduct<RealScalar> coneprod;
public:
    // n variables, d degree of cone
    // c -> R[n, 1],
    // A -> R[p, n], G -> R[d, n],
    // b -> R[p, 1], h -> R[d, 1]
    // this should probably be private, but I want to make my life easier
    optMatrix<RealScalar> A, G;
    optVector<RealScalar> b, h, c;
    int n, p, d;
    Model(const Eigen::Ref<const optVector<RealScalar>>& c,
          const Eigen::Ref<const optMatrix<RealScalar>>& A, const Eigen::Ref<const optVector<RealScalar>>& b,
          const Eigen::Ref<const optMatrix<RealScalar>>& G, const Eigen::Ref<const optVector<RealScalar>>& h,
          cone_array<RealScalar>& cones){
            this->c = c;
            this->A = A;    this->b = b;
            this->G = G;    this->h = h;
            this->coneprod = ConeProduct<RealScalar>(cones);
            this->n = c.rows(); this->p = b.rows(); this->d = h.rows();
          }
    void print_model() const;
    ConeProduct<RealScalar>& cone(){return coneprod;}
};

template<typename RealScalar>
void Model<RealScalar>::print_model() const{
    std::cout << "-------------------" << std::endl;
    std::cout << "Model parameters:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of primal variables = " << n << std::endl;
    std::cout << "Number of linear constraints = " << p << std::endl;
    std::cout << "Number of conic variables = " << d << std::endl;
    std::cout << "-------------------" << std::endl;
}

#endif