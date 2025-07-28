#ifndef LOGPERSPECEPI_PAPP_VARGA_H
#define LOGPERSPECEPI_PAPP_VARGA_H
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "cones.cpp"
#include "vectorize.cpp"

template<typename RealScalar, bool IsComplex>
class LogPerspecEpi : public Cone<RealScalar>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return types are RealMatrix and RealVector
    using MType = typename std::conditional<IsComplex, std::complex<RealScalar>, RealScalar>::type;
    using Matrix = Eigen::Matrix<MType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<MType, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealVector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    int matrix_size;
    Matrix T, X, Y, Z, I;
    Eigen::SelfAdjointEigenSolver<Matrix> eigh;
public:
    PositiveSemidefinite(int n) : matrix_size(n){
        I.setIdentity(matrix_size);
        T.setIdentity(matrix_size);
        X.setIdentity(matrix_size);
        Y.setIdentity(matrix_size);
        Z.setIdentity(matrix_size);
        // the barrier function is given by -log det Z - log det X - log det Y
        // Z = T - Plog(X, Y)
        // Plog(X, Y) = X½ log(X½ Y⁻¹ X½) X½
        this->barrier_parameter = 3 * matrix_size;
        if(IsComplex)   this->num_params = 6 * matrix_size * matrix_size;
        else    this->num_params = 3 * matrix_size * matrix_size;
    }
    RealVector point() const override{
        Vector v = Vector::Zero(num_params);
        v.head(num_params) = Vectorize::vec<RealScalar>(T);
        v.segment(num_params, num_params) = Vectorize::vec<RealScalar>(X);
        v.tail(num_params) = Vectorize::vec<RealScalar>(Y);
        return v;
    }
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        T = Vectorize::unvec<RealScalar, IsComplex>(p.head(num_params));
        X = Vectorize::unvec<RealScalar, IsComplex>(v.segment(num_params, num_params));
        Y = Vectorize::unvec<RealScalar, IsComplex>(v.tail(num_params));
        // update helper matrices
        Z = T - Plog(X, Y);
        Zinv = Z.llt().solve(I);
    }
    RealVector jacobian() const override{
        return -Vectorize::vec<RealScalar>(Pinv);
    }
    RealVector hvp(const Eigen::Ref<const RealVector>& v) const override{
        Matrix V = Vectorize::unvec<RealScalar, IsComplex>(v);
        Matrix hvp = Pinv * V * Pinv;
        return Vectorize::vec<RealScalar>(hvp);
    }
    RealVector ihvp(const Eigen::Ref<const RealVector>& v) const override{
        Matrix V = Vectorize::unvec<RealScalar, IsComplex>(v);
        Matrix ihvp = P * V * P;
        return Vectorize::vec<RealScalar>(ihvp);
    }
private:
    // helper functions
};


#endif