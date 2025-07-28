#include <iostream>
#include <XAD/XAD.hpp>
#include <mpreal.h>
#include <type_traits>

template <class T>
T f(T x0, T x1, T x2, T x3)
{
    T a = sin(x0) * cos(x1);
    T b = x2 * x3 - tan(x1 - x2);
    T c = a + 2* b;
    return c*c;
}

using DTYPE = mpfr::mpreal;

void mpfr_setup(){
    const int working_digits = 100;
    const int printing_digits = 20;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

int main()
{
    xad::sin
    std::is_compound<DTYPE>::value;
    mpfr_setup();
    // input values
    DTYPE x0 = 1.0;
    DTYPE x1 = 1.5;
    DTYPE x2 = 1.3;
    DTYPE x3 = 1.2;

    // tape and active data type for 1st order adjoint computation
    typedef xad::adj<DTYPE> mode;
    typedef mode::tape_type tape_type;
    typedef mode::active_type AD;

    // initialize tape
    tape_type tape;

    // set independent variables
    AD x0_ad = x0;
    AD x1_ad = x1;
    AD x2_ad = x2;
    AD x3_ad = x3;

    // and register them
    tape.registerInput(x0_ad);
    tape.registerInput(x1_ad);
    tape.registerInput(x2_ad);
    tape.registerInput(x3_ad);

    // start recording derivatives
    tape.newRecording();

    AD y = f(x0_ad, x1_ad, x2_ad, x3_ad);

    // register and seed adjoint of output
    tape.registerOutput(y);
    derivative(y) = 1.0;

    // compute all other adjoints
    tape.computeAdjoints();

    // output results
    std::cout << "y = " << value(y) << "\n"
              << "\nfirst order derivatives:\n"
              << "dy/dx0 = " << derivative(x0_ad) << "\n"
              << "dy/dx1 = " << derivative(x1_ad) << "\n"
              << "dy/dx2 = " << derivative(x2_ad) << "\n"
              << "dy/dx3 = " << derivative(x3_ad) << "\n";
}