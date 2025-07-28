#include <cppad/base_require.hpp>
# include <mpreal.h>

namespace CppAD {
   inline mpfr::mpreal CondExpOp(
      enum CompareOp      cop          ,
      const mpfr::mpreal&       left         ,
      const mpfr::mpreal&       right        ,
      const mpfr::mpreal&       exp_if_true  ,
      const mpfr::mpreal&       exp_if_false )
   {  return CondExpTemplate(cop, left, right, exp_if_true, exp_if_false);
   }
   CPPAD_COND_EXP_REL(mpfr::mpreal)
   inline bool EqualOpSeq(const mpfr::mpreal& x, const mpfr::mpreal& y)
   {  return x == y; }
   inline bool IdenticalCon(const mpfr::mpreal& x)
   {  return true; }
   inline bool IdenticalZero(const mpfr::mpreal& x)
   {  return (x == mpfr::mpreal(0)); }
   inline bool IdenticalOne(const mpfr::mpreal& x)
   {  return (x == mpfr::mpreal(1)); }
   inline bool IdenticalEqualCon(const mpfr::mpreal& x, const mpfr::mpreal& y)
   {  return (x == y); }
   inline int Integer(const mpfr::mpreal& x)
   {  return static_cast<int>(x); }
   CPPAD_AZMUL(mpfr::mpreal)
   inline bool GreaterThanZero(const mpfr::mpreal& x)
   {  return x > mpfr::mpreal(0); }
   inline bool GreaterThanOrZero(const mpfr::mpreal& x)
   {  return x >= mpfr::mpreal(0); }
   inline bool LessThanZero(const mpfr::mpreal& x)
   {  return x < mpfr::mpreal(0); }
   inline bool LessThanOrZero(const mpfr::mpreal& x)
   {  return x <= mpfr::mpreal(0); }
   inline bool abs_geq(const mpfr::mpreal& x, const mpfr::mpreal& y)
   {  return mpfr::fabs(x) >= mpfr::fabs(y); }
   using mpfr::acos;
   using mpfr::asin;
   using mpfr::atan;
   using mpfr::cos;
   using mpfr::cosh;
   using mpfr::exp;
   using mpfr::fabs;
   using mpfr::log;
   using mpfr::log10;
   using mpfr::sin;
   using mpfr::sinh;
   using mpfr::sqrt;
   using mpfr::tan;
   using mpfr::tanh;
   using mpfr::asinh;
   using mpfr::acosh;
   using mpfr::atanh;
   using mpfr::erf;
   using mpfr::erfc;
   using mpfr::expm1;
   using mpfr::log1p;
   inline mpfr::mpreal sign(const mpfr::mpreal& x)
   {  if( x > mpfr::mpreal(0) )
         return mpfr::mpreal(1);
      if( x == mpfr::mpreal(0) )
         return mpfr::mpreal(0);
      return mpfr::mpreal(-1);
   }
   template <> class numeric_limits<mpfr::mpreal> {
   public:
   static mpfr::mpreal min(void) 
   {  return static_cast<mpfr::mpreal>( std::numeric_limits<mpfr::mpreal>::min() ); }
   static mpfr::mpreal max(void) 
   {  return static_cast<mpfr::mpreal>( std::numeric_limits<mpfr::mpreal>::max() ); }
   static mpfr::mpreal epsilon(void) 
   {  return static_cast<mpfr::mpreal>( std::numeric_limits<mpfr::mpreal>::epsilon() ); }
   static mpfr::mpreal quiet_NaN(void) 
   {  return static_cast<mpfr::mpreal>( std::numeric_limits<mpfr::mpreal>::quiet_NaN() ); }
   static mpfr::mpreal infinity(void) 
   {  return static_cast<mpfr::mpreal>( std::numeric_limits<mpfr::mpreal>::infinity() ); }
   static int digits10(void)
   {  return std::numeric_limits<mpfr::mpreal>::digits10();}
   static int max_digits10(void)
   {  return std::numeric_limits<mpfr::mpreal>::max_digits10();}
};
    template <> struct to_string_struct<mpfr::mpreal>{
        std::string operator()(const mpfr::mpreal& value){
        std::stringstream os;
        int n_digits = 1 + std::numeric_limits<mpfr::mpreal>::digits10();
        os << std::setprecision(n_digits);
        os << value;
        return os.str();
        }
    };
}