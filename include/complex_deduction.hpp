#ifndef COMPLEX_DEDUCTION_PAPP_VARGA_H
#define COMPLEX_DEDUCTION_PAPP_VARGA_H
#include <complex>

template<typename T>
struct is_complex_t : public std::false_type {};
template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

#endif