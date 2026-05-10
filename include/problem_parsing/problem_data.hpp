#ifndef MODEL_PAPP_VARGA_H
#define MODEL_PAPP_VARGA_H
#include "cone_product.hpp"
#include "common_typedefs.hpp"
#include <iostream>
#include <Eigen/QR>

template<typename prec_type>
class ProblemData{
private:
public:
    // n variables, d degree of cone
    // c -> R[n, 1],
    // A -> R[p, n], G -> R[d, n],
    // b -> R[p, 1], h -> R[d, 1]
    // this should probably be private, but I want to make my life easier
    optMatrix<prec_type> A, G;
    optVector<prec_type> b, h, c;
    int n, p, d;
    const ConeProduct<prec_type>& cones;
    Model(const Eigen::Ref<const optVector<prec_type>>& c,
          const Eigen::Ref<const optMatrix<prec_type>>& A, const Eigen::Ref<const optVector<prec_type>>& b,
          const Eigen::Ref<const optMatrix<prec_type>>& G, const Eigen::Ref<const optVector<prec_type>>& h,
          cone_array<prec_type>& cones){
            this->c = c;
            this->A = A;    this->b = b;
            this->G = G;    this->h = h;
            this->coneprod = ConeProduct<prec_type>(cones);
            this->n = c.rows(); this->p = b.rows(); this->d = h.rows();
          }
    void print_problem_data() const;
};

template<typename prec_type>
void ProblemData<prec_type>::print_problem_data() const{
    std::cout << "-------------------" << std::endl;
    std::cout << "Model parameters:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of primal variables = " << n << std::endl;
    std::cout << "Number of linear constraints = " << p << std::endl;
    std::cout << "Number of conic variables = " << d << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "A has dimensions " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "G has dimensions " << G.rows() << " x " << G.cols() << std::endl;
    std::cout << "The rank of A is " << model.A.fullPivHouseholderQr().rank() << std::endl;
}

#endif