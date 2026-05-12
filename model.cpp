#ifndef MODEL_PAPP_VARGA_H
#define MODEL_PAPP_VARGA_H
#include "cones.cpp"
#include <Eigen/SVD>

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
          const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Vector>& b,
          const Eigen::Ref<const Matrix>& G, const Eigen::Ref<const Vector>& h,
          std::vector<std::unique_ptr<Cone<RealScalar>>>& cones){
            Eigen::BDCSVD<Matrix, Eigen::ComputeThinU> svd;
            svd.compute(A);
            this->A = svd.matrixU().leftCols(svd.rank()).transpose() * A;
            // Explain why multiplying with U^T does not change the number of non-zeros in A
            std::cout << "Non-zeros in A: " << A.cwiseGreaterOrEqual(1e-8).count() << std::endl;
            std::cout << "Non-zeros in this->A: " << A.cwiseGreaterOrEqual(1e-8).count() << std::endl;
            this->G = G;
            this->b = svd.matrixU().leftCols(svd.rank()).transpose() * b;
            this->h = h;
            this->c = c;
            this->n = this->c.rows();
            this->p = this->b.rows();
            this->d = this->h.rows();
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
    std::cout << "Number of primal variables = " << n << std::endl;
    std::cout << "Number of reduced linear constraints = " << p << std::endl;
    std::cout << "Number of conic variables = " << d << std::endl;
    std::cout << "-------------------" << std::endl;
}

#endif