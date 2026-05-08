#ifndef LOGPERSPECEPI_PAPP_VARGA_H
#define LOGPERSPECEPI_PAPP_VARGA_H
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include "cones.hpp"
#include "vectorize.hpp"
#include "common_typedefs.hpp"

template<typename prec_type>
class LogPerspecEpi : public Cone<prec_type>{
protected:
    int matrix_size;
    bool jac_updated = false;
    const prec_type eps = 1e-12;
    optMatrix<std::complex<prec_type>> T, X, Y, I;
    Eigen::SelfAdjointEigenSolver<optMatrix<std::complex<prec_type>>> eigh;
    Eigen::LDLT<optMatrix<std::complex<prec_type>>> llt;
    // precompute the jacobian
    optVector<prec_type> jac;
    // column vectors of eigenvalues of X and Y (use real vector because eigvals are real)
    optVector<prec_type> Xeig, Yeig;
    // matrices of X and Y eigenvectors
    optMatrix<std::complex<prec_type>> Xeigv, Yeigv;
    // X, Y sqrt, isqrt
    optMatrix<std::complex<prec_type>> Xsqrt, Xisqrt, Ysqrt, Yisqrt;
    // optMatrix<std::complex<prec_type>> inverses
    optMatrix<std::complex<prec_type>> Zinv, Xinv, Yinv;
    // Xtil = Yisqrt * X * Yisqrt, Ytil = Xisqrt * Y * Xisqrt
    optVector<prec_type> Xtileig, Ytileig;
    optMatrix<std::complex<prec_type>> Xtileigv, Ytileigv;
    // YsZiYs = Ysqrt * Zinv * Ysqrt, XsZiXs = Xsqrt * Zinv * Xsqrt
    optMatrix<std::complex<prec_type>> YsZiYs, XsZiXs;
public:
    LogPerspecEpi(int n);
    std::string coneName() const override {return std::string("Complex log perspective epigraph cone");}
    optVector<prec_type> point() const override;
    void updatePoint(const Eigen::Ref<const optVector<prec_type>>& p) override;
    optVector<prec_type> jacobian() override;
    optVector<prec_type> hvp(const Eigen::Ref<const optVector<prec_type>>& v) override;
private:
    // helper functions
    void computeAux();
    prec_type g1divd(prec_type a, prec_type b);
    prec_type ghat1divd(prec_type a, prec_type b);
    prec_type xghat1divd(prec_type a, prec_type b);
    prec_type g2divd(prec_type a, prec_type c, prec_type b);
    prec_type ghat2divd(prec_type a, prec_type c, prec_type b);
    prec_type xghat2divd(prec_type a, prec_type c, prec_type b);
    optMatrix<std::complex<prec_type>> Dg(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& V);
    optMatrix<std::complex<prec_type>> Dghat(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& V);
    optMatrix<std::complex<prec_type>> D2g(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& V,
        const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& W);
    optMatrix<std::complex<prec_type>> D2ghat(const Eigen::Ref<const optVector<prec_type>>& L,
        const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& U,
        const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& V,
        const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& W);
    optMatrix<std::complex<prec_type>> D2xghat(const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& V,
        const Eigen::Ref<const optMatrix<std::complex<prec_type>>>& W);
};

#endif