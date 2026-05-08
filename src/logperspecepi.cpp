#include "logperspecepi.hpp"

template<typename prec_type, bool is_complex>
LogPerspecEpi<prec_type, is_complex>::LogPerspecEpi(int n) : matrix_size(n){
    // the barrier function is given by -log det Z - log det X - log det Y
    // Z = T - Plog(X, Y)
    // Plog(X, Y) = -X½ log(X-½ Y X-½) X½
    this->barrier_parameter = 3 * matrix_size;
    this->num_params = 6 * matrix_size * matrix_size;
    T.setIdentity(matrix_size, matrix_size);
    X.setIdentity(matrix_size, matrix_size);
    Y.setIdentity(matrix_size, matrix_size);
    I.setIdentity(matrix_size, matrix_size);
    jac.resize(this->num_params);
}

template<typename prec_type, bool is_complex>
optVector<prec_type> LogPerspecEpi<prec_type, is_complex>::point() const{
    optVector<prec_type> v(this->num_params);
    v << vec<prec_type>(T), vec<prec_type>(X), vec<prec_type>(Y);
    return v;
}

template<typename prec_type, bool is_complex>
void LogPerspecEpi<prec_type, is_complex>::updatePoint(const Eigen::Ref<const optVector<prec_type>>& p){
    T = unvec<prec_type, is_complex>(p(Eigen::seqN(Eigen::fix<0>, this->num_params / 3)), matrix_size);
    X = unvec<prec_type, is_complex>(p(Eigen::seqN(this->num_params / 3, this->num_params / 3)), matrix_size);
    Y = unvec<prec_type, is_complex>(p(Eigen::placeholders::lastN(this->num_params / 3)), matrix_size);
    jac_updated = false;
}

template<typename prec_type, bool is_complex>
optVector<prec_type> LogPerspecEpi<prec_type, is_complex>::jacobian(){
    if(!jac_updated){
        // compute everything else
        computeAux();
        jac_updated = true;
    }
    return jac;
}

template<typename prec_type, bool is_complex>
optVector<prec_type> LogPerspecEpi<prec_type, is_complex>::hvp(const Eigen::Ref<const optVector<prec_type>>& v){
    if(!jac_updated){
        // compute everything else
        computeAux();
        jac_updated = true;
    }
    // it would cost too much to store the hessian in memory
    // so we compute the hvp on demand
    // There are a few parts to this. UU.adjoint() + M
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vz = unvec<prec_type, is_complex>(v(Eigen::seqN(Eigen::fix<0>, this->num_params / 3)), matrix_size);
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vx = unvec<prec_type, is_complex>(v(Eigen::seqN(this->num_params / 3, this->num_params / 3)), matrix_size);
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vy = unvec<prec_type, is_complex>(v(Eigen::placeholders::lastN(this->num_params / 3)), matrix_size);
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Tx = optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>::Zero(matrix_size, matrix_size);
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Ty = optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>::Zero(matrix_size, matrix_size);
    // let's apply M first
    // first, the purely diagonal part
    Tx.noalias() += Xinv * Vx * Xinv;
    Ty.noalias() += Yinv * Vy * Yinv;
    // then, the bilinear map part
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> YisVxYis = Yisqrt * Vx * Yisqrt;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> XisVyXis = Xisqrt * Vy * Xisqrt;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> YisVyYis = Yisqrt * Vy * Yisqrt;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> XisVxXis = Xisqrt * Vx * Xisqrt;
    Tx.noalias() += Yisqrt * D2ghat(Xtileig, Xtileigv, YsZiYs, YisVxYis) * Yisqrt;
    Tx.noalias() += Yisqrt * (Dghat(Ysqrt * Zinv * Vy * Yisqrt + Yisqrt * Vy * Zinv * Ysqrt) - D2xghat(YsZiYs, YisVyYis)) * Yisqrt;
    Ty.noalias() += Xisqrt * (Dg(Xsqrt * Zinv * Vx * Xisqrt + Xisqrt * Vx * Zinv * Xsqrt) + D2ghat(Ytileig, Ytileigv, XsZiXs, XisVxXis)) * Xisqrt;
    Ty.noalias() += Xisqrt * D2g(XsZiXs, XisVyXis) * Xisqrt;
    // let's apply the UU.adjoint() part next
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Q = Zinv * (Vz - Ysqrt * Dghat(YisVxYis) * Ysqrt - Xsqrt * Dg(XisVyXis) * Xsqrt) * Zinv;
    Tx.noalias() -= Yisqrt * Dghat(Ysqrt * Q * Ysqrt) * Yisqrt;
    Ty.noalias() -= Xisqrt * Dg(Xsqrt * Q * Xsqrt) * Xisqrt;
    optVector<prec_type> p(this->num_params);
    p << vec<prec_type>(Q), vec<prec_type>(Tx), vec<prec_type>(Ty);
    return p;
}

template<typename prec_type, bool is_complex>
void LogPerspecEpi<prec_type, is_complex>::computeAux(){
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
    eigh.compute(Yisqrt * X * Yisqrt);
    Xtileig = eigh.eigenvalues();
    Xtileigv = eigh.eigenvectors();
    eigh.compute(Xisqrt * Y * Xisqrt);
    Ytileig = eigh.eigenvalues();
    Ytileigv = eigh.eigenvectors();
    // we need to invert Z
    // remember g is -logx
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Z = T;
    Z.noalias() += Xsqrt * Ytileigv * Ytileig.array().log().matrix().asDiagonal() * Ytileigv.adjoint() * Xsqrt;
    llt.compute(Z);
    Zinv = llt.solve(I);
    YsZiYs = Ysqrt * Zinv * Ysqrt;
    XsZiXs = Xsqrt * Zinv * Xsqrt;
    // precompute the jacobian
    jac << -vec<prec_type>(Zinv),
            vec<prec_type>(Yisqrt * Dghat(YsZiYs) * Yisqrt - Xinv),
            vec<prec_type>(Xisqrt * Dg(XsZiXs) * Xisqrt - Yinv);
}


template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::g1divd(prec_type a, prec_type b){
    using std::log;
    using std::abs;
    if(abs(a - b) < prec_type(eps))  return -prec_type(1) / a;
    return -(log(a) - log(b)) / (a - b);
}
template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::ghat1divd(prec_type a, prec_type b){
    using std::log;
    using std::abs;
    if(abs(a - b) < prec_type(eps))  return log(a) + prec_type(1);
    return (a * log(a) - b * log(b)) / (a - b);
}
template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::xghat1divd(prec_type a, prec_type b){
    using std::log;
    using std::abs;
    if(abs(a - b) < prec_type(eps))  return prec_type(2) * a * log(a) + a;
    return (a * a * log(a) - b * b * log(b)) / (a - b);
}
template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::g2divd(prec_type a, prec_type c, prec_type b){
    using std::abs;
    // g[1](a, c) - g[1](c, b) / (a - b)
    // if all are equal then return 1 / (2 * x * x)
    if(abs(a - c) < prec_type(eps) and abs(c - b) < prec_type(eps))  return prec_type(1) / (prec_type(2) * a * a);
    // if a == b then
    // lim a -> b (g1(b + da, c) - g1(b, c)) / (da) = g1'(b, c) = g1'(a, c)
    // d (g(a) - g(c)) / (a - c) / da = g'(a) / (a - c) - (g(a) - g(c)) / (a - c) ^ 2 = (g'(a) - g1(a, c)) / (a - c)
    // so it's the same as swapping b and c
    if(abs(a - b) < prec_type(eps))  std::swap(b, c);
    // otherwise it will be properly handled
    return (g1divd(a, c) - g1divd(c, b)) / (a - b);
}
template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::ghat2divd(prec_type a, prec_type c, prec_type b){
    using std::abs;
    if(abs(a - c) < prec_type(eps) and abs(c - b) < prec_type(eps))  return prec_type(1) / (prec_type(2) * a);
    if(abs(a - b) < prec_type(eps))  std::swap(b, c);
    return (ghat1divd(a, c) - ghat1divd(c, b)) / (a - b);
}
template<typename prec_type, bool is_complex>
prec_type LogPerspecEpi<prec_type, is_complex>::xghat2divd(prec_type a, prec_type c, prec_type b){
    // x^2 log x
    // 2xlogx + x
    // 1/2 (2logx + 2 + 1) = logx + 3 / 2
    using std::log;
    using std::abs;
    if(abs(a - c) < prec_type(eps) and abs(c - b) < prec_type(eps))  return log(a) + prec_type(3) / prec_type(2);
    if(abs(a - b) < prec_type(eps))  std::swap(b, c);
    return (xghat1divd(a, c) - xghat1divd(c, b)) / (a - b);
}
template<typename prec_type, bool is_complex>
optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> LogPerspecEpi<prec_type, is_complex>::Dg(const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& V){
    // -log(x)
    // compute first divided differences
    // Dg is always called with Ytil
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> F(Ytileig.size(), Ytileig.size());
    using std::log;
    for(int i = 0; i < F.rows(); ++i){
        for(int j = 0; j < F.cols(); ++j){
            F(i, j) = g1divd(Ytileig(i), Ytileig(j));
        }
    }
    return Ytileigv * (F.cwiseProduct(Ytileigv.adjoint() * V * Ytileigv)) * Ytileigv.adjoint();
}
template<typename prec_type, bool is_complex>
optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> LogPerspecEpi<prec_type, is_complex>::Dghat(const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& V){
    // xlog(x)
    // compute first divided differences
    // Dghat is always called with Xtil
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> F(Xtileig.size(), Xtileig.size());
    using std::log;
    for(int i = 0; i < F.rows(); ++i){
        for(int j = 0; j < F.cols(); ++j){
            F(i, j) = ghat1divd(Xtileig(i), Xtileig(j));
        }
    }
    return Xtileigv * (F.cwiseProduct(Xtileigv.adjoint() * V * Xtileigv)) * Xtileigv.adjoint();
}
template<typename prec_type, bool is_complex>
optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> LogPerspecEpi<prec_type, is_complex>::D2g(const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& V, const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& W){
    // -log(x)
    // compute second divided differences
    // always called with Ytil
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> C = optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>::Zero(Ytileig.size(), Ytileig.size());
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vu = Ytileigv.adjoint() * V * Ytileigv;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Wu = Ytileigv.adjoint() * W * Ytileigv;
    // we need to sum over k
    for(int k = 0; k < Ytileig.size(); ++k){
        // for the current k, we calculate second divided differences
        optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> F2k(Ytileig.size(), Ytileig.size());
        for(int i = 0; i < F2k.rows(); ++i){
            for(int j = 0; j < F2k.cols(); ++j){
                F2k(i, j) = g2divd(Ytileig(i), Ytileig(k), Ytileig(j));
            }
        }
        C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
    }
    return Ytileigv * C * Ytileigv.adjoint();
}
template<typename prec_type, bool is_complex>
optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> LogPerspecEpi<prec_type, is_complex>::D2ghat(const Eigen::Ref<const optVector<prec_type>>& L, const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& U, const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& V, const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& W){
    // xlog(x)
    // compute second divided differences
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> C = optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>::Zero(L.size(), L.size());
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vu = U.adjoint() * V * U;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Wu = U.adjoint() * W * U;
    // we need to sum over k
    for(int k = 0; k < L.size(); ++k){
        // for the current k, we calculate second divided differences
        optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> F2k(L.size(), L.size());
        for(int i = 0; i < F2k.rows(); ++i){
            for(int j = 0; j < F2k.cols(); ++j){
                F2k(i, j) = ghat2divd(L(i), L(k), L(j));
            }
        }
        C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
    }
    return U * C * U.adjoint();
}
template<typename prec_type, bool is_complex>
optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> LogPerspecEpi<prec_type, is_complex>::D2xghat(const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& V, const Eigen::Ref<const optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>>& W){
    // x^2 log(x)
    // compute second divided differences
    // always called with Xtil
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> C = optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>>::Zero(Xtileig.size(), Xtileig.size());
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Vu = Xtileigv.adjoint() * V * Xtileigv;
    optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> Wu = Xtileigv.adjoint() * W * Xtileigv;
    // we need to sum over k
    for(int k = 0; k < Xtileig.size(); ++k){
        // for the current k, we calculate second divided differences
        optMatrix<std::conditional_t<is_complex, std::complex<prec_type>, prec_type>> F2k(Xtileig.size(), Xtileig.size());
        for(int i = 0; i < F2k.rows(); ++i){
            for(int j = 0; j < F2k.cols(); ++j){
                F2k(i, j) = xghat2divd(Xtileig(i), Xtileig(k), Xtileig(j));
            }
        }
        C += F2k.cwiseProduct(Vu.col(k) * Wu.row(k) + Wu.col(k) * Vu.row(k));
    }
    return Xtileigv * C * Xtileigv.adjoint();
}


template class LogPerspecEpi<double, false>;
template class LogPerspecEpi<double, true>;