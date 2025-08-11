#ifndef LOGPERSPECEPI_PAPP_VARGA_H
#define LOGPERSPECEPI_PAPP_VARGA_H
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "cones.cpp"
#include "vectorize.cpp"

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
// class MatrixM : public Eigen::EigenBase<MatrixM> {
//  public:
//   // Required typedefs, constants, and method:
//   typedef double Scalar;
//   typedef double RealScalar;
//   typedef int StorageIndex;
//   enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };
 
//   Index rows() const { return mp_mat->rows(); }
//   Index cols() const { return mp_mat->cols(); }
 
//   template <typename Rhs>
//   Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
//     return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
//   }
// };

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
    bool jac_updated = false;
    const RealScalar eps = 1e-12;
    Matrix T, X, Y, I;
    Eigen::SelfAdjointEigenSolver<Matrix> eigh;
    Eigen::LDLT<Matrix> llt;
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
    // YsZiYs = Ysqrt * Zinv * Ysqrt, XsZiXs = Xsqrt * Zinv * Xsqrt
    Matrix YsZiYs, XsZiXs;
    // Matrix specialization for CG
    // MatrixM M;
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
    }
    RealVector point() const override{
        RealVector v(this->num_params);
        v << Vectorize::vec<RealScalar>(T), Vectorize::vec<RealScalar>(X), Vectorize::vec<RealScalar>(Y);
        return v;
    }
    void updatePoint(const Eigen::Ref<const RealVector>& p) override{
        T = Vectorize::unvec<RealScalar, IsComplex>(p.head(this->num_params / 3));
        X = Vectorize::unvec<RealScalar, IsComplex>(p.segment(this->num_params / 3, this->num_params / 3));
        Y = Vectorize::unvec<RealScalar, IsComplex>(p.tail(this->num_params / 3));
        jac_updated = false;
    }
    RealVector jacobian() override{
        if(!jac_updated){
            // compute everything else
            computeAux();
            jac_updated = true;
        }
        return jac;
    }
    RealVector hvp(const Eigen::Ref<const RealVector>& v) override{
        if(!jac_updated){
            // compute everything else
            computeAux();
            jac_updated = true;
        }
        // it would cost too much to store the hessian in memory
        // so we compute the hvp on demand
        // There are a few parts to this. UU.adjoint() + M
        Matrix Vz = Vectorize::unvec<RealScalar, IsComplex>(v.head(this->num_params / 3));
        Matrix Vx = Vectorize::unvec<RealScalar, IsComplex>(v.segment(this->num_params / 3, this->num_params / 3));
        Matrix Vy = Vectorize::unvec<RealScalar, IsComplex>(v.tail(this->num_params / 3));
        Matrix Tx = Matrix::Zero(matrix_size, matrix_size);
        Matrix Ty = Matrix::Zero(matrix_size, matrix_size);
        // let's apply M first
        // first, the purely diagonal part
        Tx.noalias() += Xinv * Vx * Xinv;
        Ty.noalias() += Yinv * Vy * Yinv;
        // then, the bilinear map part
        Matrix YisVxYis = Yisqrt * Vx * Yisqrt;
        Matrix XisVyXis = Xisqrt * Vy * Xisqrt;
        Matrix YisVyYis = Yisqrt * Vy * Yisqrt;
        Matrix XisVxXis = Xisqrt * Vx * Xisqrt;
        Tx.noalias() += Yisqrt * D2ghat(Xtil, YsZiYs, YisVxYis) * Yisqrt;
        Tx.noalias() += Yisqrt * (Dghat(Xtil, Ysqrt * Zinv * Vy * Yisqrt + Yisqrt * Vy * Zinv * Ysqrt) - D2xghat(Xtil, YsZiYs, YisVyYis)) * Yisqrt;
        Ty.noalias() += Xisqrt * (Dg(Ytil, Xsqrt * Zinv * Vx * Xisqrt + Xisqrt * Vx * Zinv * Xsqrt) + D2ghat(Ytil, XsZiXs, XisVxXis)) * Xisqrt;
        Ty.noalias() += Xisqrt * D2g(Ytil, XsZiXs, XisVyXis) * Xisqrt;
        // let's apply the UU.adjoint() part next
        Matrix Q = Zinv * (Vz - Ysqrt * Dghat(Xtil, YisVxYis) * Ysqrt - Xsqrt * Dg(Ytil, XisVyXis) * Xsqrt) * Zinv;
        Tx.noalias() -= Yisqrt * Dghat(Xtil, Ysqrt * Q * Ysqrt) * Yisqrt;
        Ty.noalias() -= Xisqrt * Dg(Ytil, Xsqrt * Q * Xsqrt) * Xisqrt;
        RealVector p(this->num_params);
        p << Vectorize::vec<RealScalar>(Q), Vectorize::vec<RealScalar>(Tx), Vectorize::vec<RealScalar>(Ty);
        return p;
    }
    // RealVector ihvp(const Eigen::Ref<const RealVector>& v) const override{
    //     if(!jac_updated){
    //         // compute everything else
    //         computeAux();
    //         jac_updated = true;
    //     }
    //     Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    //     cg.compute(A);
    // }
private:
    friend class MatrixM;
    // helper functions
    void computeAux(){
        // Z = T - Plog(X, Y)
        // Plog(X, Y) = -X½ log(X-½ Y X-½) X½
        // we need the eigendecomposition of X and Y
        eigh.compute(X);
        Xeig = eigh.eigenvalues();
        Xeigv = eigh.eigenvectors();
        eigh.compute(Y);
        Yeig = eigh.eigenvalues();
        Yeigv = eigh.eigenvectors();
        // Calculate the inv, sqrt, isqrt matrices
        Xinv.noalias() = Xeigv * Xeig.cwiseInverse().asDiagonal() * Xeigv.adjoint();
        Yinv.noalias() = Yeigv * Yeig.cwiseInverse().asDiagonal() * Yeigv.adjoint();
        Xsqrt.noalias() = Xeigv * Xeig.cwiseSqrt().asDiagonal() * Xeigv.adjoint();
        Xisqrt.noalias() = Xeigv * Xeig.cwiseSqrt().cwiseInverse().asDiagonal() * Xeigv.adjoint();
        Ysqrt.noalias() = Yeigv * Yeig.cwiseSqrt().asDiagonal() * Yeigv.adjoint();
        Yisqrt.noalias() = Yeigv * Yeig.cwiseSqrt().cwiseInverse().asDiagonal() * Yeigv.adjoint();
        // Calculate the til matrices
        Xtil.noalias() = Yisqrt * X * Yisqrt;
        Ytil.noalias() = Xisqrt * Y * Xisqrt;
        // we need to invert Z
        // remember g is -logx
        Matrix Z = T;
        eigh.compute(Ytil);
        Z.noalias() += Xsqrt * eigh.eigenvectors() * eigh.eigenvalues().array().log().matrix().asDiagonal() * eigh.eigenvectors().adjoint() * Xsqrt;
        llt.compute(Z);
        Zinv = llt.solve(I);
        YsZiYs = Ysqrt * Zinv * Ysqrt;
        XsZiXs = Xsqrt * Zinv * Xsqrt;
        // precompute the jacobian
        jac << -Vectorize::vec<RealScalar>(Zinv), Vectorize::vec<RealScalar>(Yisqrt * Dghat(Xtil, YsZiYs) * Yisqrt - Xinv), Vectorize::vec<RealScalar>(Xisqrt * Dg(Ytil, XsZiXs) * Xisqrt - Yinv);
    }
    // we need frechet here...
    RealScalar g1divd(RealScalar a, RealScalar b){
        using std::log;
        using std::abs;
        if(abs(a - b) < RealScalar(eps))  return -RealScalar(1) / a;
        return -(log(a) - log(b)) / (a - b);
    }
    RealScalar ghat1divd(RealScalar a, RealScalar b){
        using std::log;
        using std::abs;
        if(abs(a - b) < RealScalar(eps))  return log(a) + RealScalar(1);
        return (a * log(a) - b * log(b)) / (a - b);
    }
    RealScalar xghat1divd(RealScalar a, RealScalar b){
        using std::log;
        using std::abs;
        if(abs(a - b) < RealScalar(eps))  return RealScalar(2) * a * log(a) + a;
        return (a * a * log(a) - b * b * log(b)) / (a - b);
    }
    RealScalar g2divd(RealScalar a, RealScalar c, RealScalar b){
        using std::abs;
        // g[1](a, c) - g[1](c, b) / (a - b)
        // if all are equal then return 1 / (2 * x * x)
        if(abs(a - c) < RealScalar(eps) and abs(c - b) < RealScalar(eps))  return RealScalar(1) / (RealScalar(2) * a * a);
        // if a == b then
        // lim a -> b (g1(b + da, c) - g1(b, c)) / (da) = g1'(b, c) = g1'(a, c)
        // d (g(a) - g(c)) / (a - c) / da = g'(a) / (a - c) - (g(a) - g(c)) / (a - c) ^ 2 = (g'(a) - g1(a, c)) / (a - c)
        // so it's the same as swapping b and c
        if(abs(a - b) < RealScalar(eps))  std::swap(b, c);
        // otherwise it will be properly handled
        return (g1divd(a, c) - g1divd(c, b)) / (a - b);
    }
    RealScalar ghat2divd(RealScalar a, RealScalar c, RealScalar b){
        using std::abs;
        if(abs(a - c) < RealScalar(eps) and abs(c - b) < RealScalar(eps))  return RealScalar(1) / (RealScalar(2) * a);
        if(abs(a - b) < RealScalar(eps))  std::swap(b, c);
        return (ghat1divd(a, c) - ghat1divd(c, b)) / (a - b);
    }
    RealScalar xghat2divd(RealScalar a, RealScalar c, RealScalar b){
        // x^2 log x
        // 2xlogx + x
        // 1/2 (2logx + 2 + 1) = logx + 3 / 2
        using std::log;
        using std::abs;
        if(abs(a - c) < RealScalar(eps) and abs(c - b) < RealScalar(eps))  return log(a) + RealScalar(3) / RealScalar(2);
        if(abs(a - b) < RealScalar(eps))  std::swap(b, c);
        return (xghat1divd(a, c) - xghat1divd(c, b)) / (a - b);
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
            C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
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
            C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
        }
        return U * C * U.adjoint();
    }
    Matrix D2xghat(const Eigen::Ref<const Matrix>& A, const Eigen::Ref<const Matrix>& V, const Eigen::Ref<const Matrix>& W){
        // x^2 log(x)
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
                    F2k(i, j) = xghat2divd(L(i), L(k), L(j));
                }
            }
            C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
        }
        return U * C * U.adjoint();
    }
};

// IHVP

// // Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
// namespace Eigen {
// namespace internal {
 
// template <typename Rhs>
// struct generic_product_impl<LPEIHVP, Rhs, SparseShape, DenseShape,
//                             GemvProduct>  // GEMV stands for matrix-vector
//     : generic_product_impl_base<LPEIHVP, Rhs, generic_product_impl<LPEIHVP, Rhs> > {
//   typedef typename Product<LPEIHVP, Rhs>::Scalar Scalar;
 
//   template <typename Dest>
//   static void scaleAndAddTo(Dest& dst, const LPEIHVP& lhs, const Rhs& rhs, const Scalar& alpha) {
//     // This method should implement "dst += alpha * lhs * rhs" inplace,
//     // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
//     eigen_assert(alpha == Scalar(1) && "scaling is not implemented");
//     EIGEN_ONLY_USED_FOR_DEBUG(alpha);
 

//   }
// };
 
// }  // namespace internal
// }  /

#endif