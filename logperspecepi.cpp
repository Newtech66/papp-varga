#ifndef LOGPERSPECEPI_PAPP_VARGA_H
#define LOGPERSPECEPI_PAPP_VARGA_H
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "cones.cpp"
#include "vectorize.cpp"

template<typename RealScalar, bool IsComplex>
class LogPerspecEpi : public Cone<RealScalar>{
    // internally, Matrix and Vector types are used
    // externally, the argument and return type is RealVector
    using MType = typename std::conditional<IsComplex, std::complex<RealScalar>, RealScalar>::type;
    using Matrix = Eigen::Matrix<MType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<MType, Eigen::Dynamic>;
    using RealVector = Eigen::Vector<RealScalar, Eigen::Dynamic>;
protected:
    int matrix_size;
    Matrix T, X, Y, I;
    Eigen::SelfAdjointEigenSolver<Matrix> eigh;
    Eigen::LLT<Matrix> llt;
    // precompute the jacobian
    RealVector jac;
    // column vectors of eigenvalues of X and Y (use real vector because eigvals are real)
    RealVector Xeig, Yeig;
    // matrices of X and Y eigenvectors
    Matrix Xeigv, Yeigv;
    // X, Y sqrt, isqrt
    Matrix Xsqrt, Xisqrt, Ysqrt, Yisqrt;
    // Matrix inverses
    Matrix Zinv, Xinv, Yinv;
    // Xtil = Yisqrt * X * Yisqrt, Ytil = Xisqrt * Y * Xisqrt
    Matrix Xtil, Ytil;
public:
    LogPerspecEpi(int n) : matrix_size(n){
        // the barrier function is given by -log det Z - log det X - log det Y
        // Z = T - Plog(X, Y)
        // Plog(X, Y) = -X½ log(X-½ Y X-½) X½
        this->barrier_parameter = 3 * matrix_size;
        if(IsComplex)   this->num_params = 6 * matrix_size * matrix_size;
        else    this->num_params = 3 * matrix_size * matrix_size;
        T.setIdentity(matrix_size, matrix_size);
        X.setIdentity(matrix_size, matrix_size);
        Y.setIdentity(matrix_size, matrix_size);
        I.setIdentity(matrix_size, matrix_size);
        jac.resize(this->num_params);
        // compute everything else
        computeAux();
    }
    RealVector point() const override{
        RealVector v(this->num_params);
        v << Vectorize::vec<RealScalar>(T), Vectorize::vec<RealScalar>(X), Vectorize::vec<RealScalar>(Y);
        return v;
    }
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        T = Vectorize::unvec<RealScalar, IsComplex>(p.head(this->num_params));
        X = Vectorize::unvec<RealScalar, IsComplex>(p.segment(this->num_params, this->num_params));
        Y = Vectorize::unvec<RealScalar, IsComplex>(p.tail(this->num_params));
        // compute everything else
        computeAux();
    }
    RealVector jacobian() const override{return jac;}
    RealVector hvp(const Eigen::Ref<const RealVector>& v) override{
        // it would cost too much to store the hessian in memory
        // so we compute the hvp on demand
        // There are a few parts to this. UU.adjoint() + M
        Matrix Vx = Vectorize::unvec<RealScalar, IsComplex>(v.head(this->num_params / 3));
        Matrix Vy = Vectorize::unvec<RealScalar, IsComplex>(v.segment(this->num_params / 3, this->num_params / 3));
        Matrix Vz = Vectorize::unvec<RealScalar, IsComplex>(v.tail(this->num_params / 3));
        Matrix Tx = Matrix::Zero(matrix_size, matrix_size);
        Matrix Ty = Matrix::Zero(matrix_size, matrix_size);
        Matrix Tz = Matrix::Zero(matrix_size, matrix_size);
        // let's apply M first
        // first, the purely diagonal part
        Tx += Xinv * Vx * Xinv;
        Ty += Yinv * Vy * Yinv;
        // then, the bilinear map part
        Tx += Dxx(Zinv, Vx) + Dxy(Zinv, Vy);
        Ty += Dxy(Zinv, Vx) + Dyy(Zinv, Vy);
        // let's apply the UU.adjoint() part next
        Matrix Q = Zinv * (Vz + Dx(Vx) + Dy(Vy)) * Zinv;
        Tz += Q;
        Tx += Dx(Q);
        Ty += Dy(Q);
        RealVector p(this->num_params);
        p << Vectorize::vec<RealScalar>(Tz), Vectorize::vec<RealScalar>(Tx), Vectorize::vec<RealScalar>(Ty);
        return p;
    }
    // RealVector ihvp(const Eigen::Ref<const RealVector>& v) const override{
    // }
private:
    // helper functions
    Matrix Dx(const Eigen::Ref<const Matrix>& V){
        // remember ghat is xlog(x)
        Matrix Vtil = Yisqrt * V * Yisqrt;
        return Ysqrt * Dghat(Xtil, Vtil) * Ysqrt;
    }
    Matrix Dy(const Eigen::Ref<const Matrix>& V){
        // remember g is -log(x)
        Matrix Vtil = Xisqrt * V * Xisqrt;
        return Xsqrt * Dg(Ytil, Vtil) * Xsqrt;
    }
    Matrix Dxx(const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        Matrix Vtil = Yisqrt * V * Yisqrt;
        Matrix Wtil = Yisqrt * W * Yisqrt;
        return Ysqrt * D2ghat(Xtil, Vtil, Wtil) * Ysqrt;
    }
    Matrix Dxy(const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        // remember h is -xlogx
        // which is -ghat so... replace -D2h with + D2ghat
        Matrix Vtil = Xisqrt * V * Xisqrt;
        Matrix Wtil = Xisqrt * W * Xisqrt;
        return Xsqrt * (Vtil * Dg(Ytil, Wtil) + Dg(Ytil, Wtil) * Vtil + D2ghat(Ytil, Vtil, Wtil)) * Xsqrt;
    }
    Matrix Dyy(const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        Matrix Vtil = Xisqrt * V * Xisqrt;
        Matrix Wtil = Xisqrt * W * Xisqrt;
        return Xsqrt * D2g(Ytil, Vtil, Wtil) * Xsqrt;
    }
    void computeAux(){
        // Z = T - Plog(X, Y)
        // Plog(X, Y) = X½ log(X½ Y⁻¹ X½) X½
        // we need the eigendecomposition of X and Y
        eigh.compute(X);
        Xeig = eigh.eigenvalues();
        Xeigv = eigh.eigenvectors();
        eigh.compute(Y);
        Yeig = eigh.eigenvalues();
        Yeigv = eigh.eigenvectors();
        // Calculate the inv, sqrt, isqrt matrices
        Xinv = Xeigv * Xeig.cwiseInverse().asDiagonal() * Xeigv.adjoint();
        Xsqrt = Xeigv * Xeig.cwiseSqrt().asDiagonal() * Xeigv.adjoint();
        Xisqrt = Xeigv * Xeig.cwiseSqrt().cwiseInverse().asDiagonal() * Xeigv.adjoint();
        Ysqrt = Yeigv * Yeig.cwiseSqrt().asDiagonal() * Yeigv.adjoint();
        Yisqrt = Yeigv * Yeig.cwiseSqrt().cwiseInverse().asDiagonal() * Yeigv.adjoint();
        // Calculate the til matrices
        Xtil = Yisqrt * X * Yisqrt;
        Ytil = Xisqrt * Y * Xisqrt;
        // we need to invert Z
        // remember g is -logx
        Matrix Z = T + Xsqrt * g(Ytil) * Xsqrt;
        llt.compute(Z);
        Zinv = llt.solve(I);
        // precompute the jacobian
        jac << -Vectorize::vec<RealScalar>(Zinv), Vectorize::vec<RealScalar>(Dx(Zinv) - Xinv), Vectorize::vec<RealScalar>(Dy(Zinv) - Yinv);
    }
    Matrix g(const Eigen::Ref<const Matrix>& V){
        // -log(x)
        eigh.compute(V);
        return - eigh.eigenvectors() * eigh.eigenvalues().cwiseInverse().asDiagonal() * eigh.eigenvectors().adjoint();
    }
    // we need frechet here...
    RealScalar g1divd(RealScalar a, RealScalar b){
        using std::log;
        if(a == b)  return -RealScalar(1) / a;
        return -(log(a) - log(b)) / (a - b);
    }
    RealScalar ghat1divd(RealScalar a, RealScalar b){
        using std::log;
        if(a == b)  return log(a) + RealScalar(1);
        return (a * log(a) - b * log(b)) / (a - b);
    }
    RealScalar g2divd(RealScalar a, RealScalar c, RealScalar b){
        // g[1](a, c) - g[1](c, b) / (a - b)
        // if all are equal then return 1 / (2 * x * x)
        if(a == c and c == b)  return RealScalar(1) / (RealScalar(2) * a * a);
        // if a == b then
        // lim a -> b (g1(b + da, c) - g1(b, c)) / (da) = g1'(b, c) = g1'(a, c)
        // d (g(a) - g(c)) / (a - c) / da = g'(a) / (a - c) - (g(a) - g(c)) / (a - c) ^ 2 = (g'(a) - g1(a, c)) / (a - c)
        // so it's the same as swapping b and c
        if(a == b)  std::swap(b, c);
        // otherwise it will be properly handled
        return (g1divd(a, c) - g1divd(c, b)) / (a - b);
    }
    RealScalar ghat2divd(RealScalar a, RealScalar c, RealScalar b){
        if(a == c and c == b)  return RealScalar(1) / (RealScalar(2) * a);
        if(a == b)  std::swap(b, c);
        return (ghat1divd(a, c) - ghat1divd(c, b)) / (a - b);
    }
    Matrix Dg(const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& V){
        // -log(x)
        // compute first divided differences
        eigh.compute(A);
        Matrix U = eigh.eigenvectors();
        RealVector L = eigh.eigenvalues();
        Matrix F(L.size(), L.size());
        using std::log;
        for(int i = 0; i < F.rows(); ++i){
            for(int j = 0; j < F.cols(); ++j){
                F(i, j) = g1divd(L(i), L(j));
            }
        }
        return U * (F.cwiseProduct(U.adjoint() * V * U)) * U.adjoint();
    }
    Matrix Dghat(const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& V){
        // xlog(x)
        // compute first divided differences
        eigh.compute(A);
        Matrix U = eigh.eigenvectors();
        RealVector L = eigh.eigenvalues();
        Matrix F(L.size(), L.size());
        using std::log;
        for(int i = 0; i < F.rows(); ++i){
            for(int j = 0; j < F.cols(); ++j){
                F(i, j) = ghat1divd(L(i), L(j));
            }
        }
        return U * (F.cwiseProduct(U.adjoint() * V * U)) * U.adjoint();
    }
    Matrix D2g(const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        // -log(x)
        // compute second divided differences
        eigh.compute(A);
        Matrix U = eigh.eigenvectors();
        RealVector L = eigh.eigenvalues();
        Matrix C = Matrix::Zero(L.size(), L.size());
        Matrix Vu = U.adjoint() * V * U;
        Matrix Wu = U.adjoint() * W * U;
        // we need to sum over k
        for(int k = 0; k < L.size(); ++k){
            // for the current k, we calculate second divided differences
            Matrix F2k(L.size(), L.size());
            for(int i = 0; i < F2k.rows(); ++i){
                for(int j = 0; j < F2k.cols(); ++j){
                    F2k(i, j) = g2divd(L(i), L(k), L(j));
                }
            }
            C += F2k.cwiseProduct(Vu.col(k) * Vu.row(k) + Wu.col(k) * Vu.row(k));
        }
        return U * C * U.adjoint();
    }
    Matrix D2ghat(const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        // xlog(x)
        // compute second divided differences
        eigh.compute(A);
        Matrix U = eigh.eigenvectors();
        RealVector L = eigh.eigenvalues();
        Matrix C = Matrix::Zero(L.size(), L.size());
        Matrix Vu = U.adjoint() * V * U;
        Matrix Wu = U.adjoint() * W * U;
        // we need to sum over k
        for(int k = 0; k < L.size(); ++k){
            // for the current k, we calculate second divided differences
            Matrix F2k(L.size(), L.size());
            for(int i = 0; i < F2k.rows(); ++i){
                for(int j = 0; j < F2k.cols(); ++j){
                    F2k(i, j) = ghat2divd(L(i), L(k), L(j));
                }
            }
            C += F2k.cwiseProduct(Vu.col(k) * Vu.row(k) + Wu.col(k) * Vu.row(k));
        }
        return U * C * U.adjoint();
    }
};

#endif