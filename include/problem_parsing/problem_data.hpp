#ifndef MODEL_PAPP_VARGA_H
#define MODEL_PAPP_VARGA_H
#include "cone_product.hpp"
#include "common_typedefs.hpp"
#include <iostream>
#include <Eigen/QR>

template<typename prec_type>
class ProblemData{
public:
    // n variables, d degree of cone
    // c -> R[n, 1],
    // A -> R[p, n], G -> R[d, n],
    // b -> R[p, 1], h -> R[d, 1]
    // this should probably be private, but I want to make my life easier
    optMatrix<prec_type> A, G;
    optVector<prec_type> b, h, c;
    int n_privar, n_lincon, n_convar;
    const ConeProduct<prec_type> cones;
    ProblemData(const Eigen::Ref<const optVector<prec_type>>& c,
        const Eigen::Ref<const optMatrix<prec_type>>& A, const Eigen::Ref<const optVector<prec_type>>& b, 
        const Eigen::Ref<const optMatrix<prec_type>>& G, const Eigen::Ref<const optVector<prec_type>>& h){
            // Cones need to be added directly through cones.addCone()
            this->c = c;
            this->A = A;    this->b = b;
            this->G = G;    this->h = h;
            this->n_privar = c.rows(); this->n_lincon = b.rows(); this->n_convar = h.rows();
          }
    void print_problem_data() const{
    std::cout << "-------------------" << std::endl;
    std::cout << "Model parameters:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "Number of primal variables = " << n_privar << std::endl;
    std::cout << "Number of linear constraints = " << n_lincon << std::endl;
    std::cout << "Number of conic variables = " << n_convar << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "A has dimensions " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "G has dimensions " << G.rows() << " x " << G.cols() << std::endl;
}
};

#endif