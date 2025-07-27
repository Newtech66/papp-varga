#ifndef POSSEMIDEFINITE_PAPP_VARGA_H
#define POSSEMIDEFINITE_PAPP_VARGA_H
#include <complex>
#include <type_traits>
#include <Eigen/Core>
#include "cones.cpp"

// Right now this only supports real stuff
template<typename RealScalar, bool IsComplex = false>
class PositiveSemidefinite : public Cone<RealScalar>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return types are RealMatrix and RealVector
    using MType = typename std::conditional<IsComplex, std::complex<RealScalar>, RealScalar>::type;
    using Matrix = Eigen::Matrix<MType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<MType, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealVector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    int matrix_size;
    Matrix P, Pinv, I;
public:
    PositiveSemidefinite(int n) : matrix_size(n){
        I = Matrix::Identity(matrix_size, matrix_size);
        P = I;
        Pinv = I;
        this->barrier_parameter = matrix_size;
        if(IsComplex)   this->num_params = 2 * matrix_size * matrix_size;
        else    this->num_params = matrix_size * matrix_size;
    }
    RealVector point() const override{return P.template reshaped<Eigen::RowMajor>();}
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        P = p.template reshaped<Eigen::RowMajor>(matrix_size, matrix_size);
        Pinv = P.llt().solve(I);
    }
    RealVector jacobian() const override{return -Pinv.template reshaped<Eigen::RowMajor>();}
    RealVector hvp(const Eigen::Ref<const RealVector>& v) const override{
        Matrix V = v.template reshaped<Eigen::RowMajor>(matrix_size, matrix_size);
        Matrix hvp = Pinv * V * Pinv;
        return hvp.template reshaped<Eigen::RowMajor>();
    }
    RealVector ihvp(const Eigen::Ref<const RealVector>& v) const override{
        Matrix V = v.template reshaped<Eigen::RowMajor>(matrix_size, matrix_size);
        Matrix ihvp = P * V * P;
        return ihvp.template reshaped<Eigen::RowMajor>();
    }
};

#endif