#ifndef MODEL_PAPP_VARGA_H
#define MODEL_PAPP_VARGA_H
#include <iostream>
#include <Eigen/Core>
#include "cones.cpp"
#include "vectorization.cpp"

template<typename RealScalar>
class Model{
    using Matrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
private:
    ConeProduct<RealScalar> coneprod;
public:
    // n variables, d degree of cone
    // c -> R[n, 1],
    // A -> R[p, n], G -> R[d, n],
    // b -> R[p, 1], h -> R[d, 1]
    // this should probably be private, but I want to make my life easier
    Matrix A, G;
    Vector b, h, c;
    int n, p, d;
    Model(const Eigen::Ref<const Vector>& c,
          const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& G,
          const Eigen::Ref<const Vector>& b, const Eigen::Ref<const Vector>& h,
          std::vector<std::unique_ptr<Cone<RealScalar>>>& cones){
            // I can be passed
            this->A = A;
            this->G = G;
            this->b = b;
            this->h = h;
            this->c = c;
            this->n = c.rows();
            this->p = b.rows();
            this->d = h.rows();
            this->coneprod = ConeProduct<RealScalar>(cones);
          }
    void print_model() const;
    ConeProduct<RealScalar>& cone(){return coneprod;}
};

template<typename RealScalar>
void Model<RealScalar>::print_model() const{
    std::cout << "-------------------" << std::endl;
    std::cout << "Model parameters:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "A =" << std::endl;
    std::cout << A << std::endl;
    std::cout << "G =" << std::endl;
    std::cout << G << std::endl;
    std::cout << "b =" << std::endl;
    std::cout << b << std::endl;
    std::cout << "h =" << std::endl;
    std::cout << h << std::endl;
    std::cout << "c =" << std::endl;
    std::cout << c << std::endl;
}

#endif