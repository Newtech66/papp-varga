#ifndef POSSEMIDEFINITE_PAPP_VARGA_H
#define POSSEMIDEFINITE_PAPP_VARGA_H
#include "vectorize.cpp"
#include "cones.cpp"

template<typename RealScalar, bool IsComplex>
class PositiveSemidefinite : public Cone<RealScalar>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return types are RealMatrix and RealVector
    using MType = typename std::conditional<IsComplex, std::complex<RealScalar>, RealScalar>::type;
    using Matrix = Eigen::Matrix<MType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<MType, Eigen::Dynamic>;
    using RealMatrix = Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using RealVector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    inline static const std::string cone_name{"Positive Semidefinite Cone"};
    int matrix_size;
    Matrix P, Pinv, I;
public:
    PositiveSemidefinite(const int n) : matrix_size(n){
        I.setIdentity(matrix_size, matrix_size);
        P.setIdentity(matrix_size, matrix_size);
        Pinv.setIdentity(matrix_size, matrix_size);
        this->barrier_parameter = matrix_size;
        if(IsComplex)   this->num_params = 2 * matrix_size * matrix_size;
        else    this->num_params = matrix_size * matrix_size;
    }
    RealVector point() const override{
        return Vectorize::vec<RealScalar>(P);
    }
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        P = Vectorize::unvec<RealScalar, IsComplex>(p);
        Pinv = P.llt().solve(I);
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
};

template<typename RealScalar>
class DiagonalPositiveSemidefinite : public Cone<RealScalar>{
    using Matrix = Eigen::DiagonalMatrix<RealScalar, Eigen::Dynamic>;
    using Vector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    inline static const std::string cone_name{"Diagonal Positive Semidefinite Cone"};
    int matrix_size;
    Matrix P, Pinv;
public:
    DiagonalPositiveSemidefinite(const int n) : matrix_size(n){
        P.setIdentity(matrix_size);
        Pinv.setIdentity(matrix_size);
        this->barrier_parameter = matrix_size;
        this->num_params = matrix_size;
    }
    Vector point() const override{
        return P.diagonal();
    }
    void updatePoint(const Eigen::Ref<const Vector>& p) override{
        P = p.asDiagonal();
        Pinv = P.inverse();
    }
    Vector jacobian() const override{
        return -Pinv.diagonal();
    }
    Vector hvp(const Eigen::Ref<const Vector>& v) const override{
        Matrix V = v.asDiagonal();
        Matrix hvp = Pinv * V * Pinv;
        return hvp.diagonal();
    }
    Vector ihvp(const Eigen::Ref<const Vector>& v) const override{
        Matrix V = v.asDiagonal();
        Matrix ihvp = P * V * P;
        return ihvp.diagonal();
    }
};

#endif