
# include "cppad_mpreal.cpp"
# include <cppad/example/cppad_eigen.hpp>
# include <cppad/speed/det_by_minor.hpp>
# include <Eigen/Dense>

bool eigen_det(void)
{  bool ok = true;
   using CppAD::AD;
   using CppAD::NearEqual;
   using Eigen::Matrix;
   using Eigen::Dynamic;
   using Eigen::Index;
   //
   typedef Matrix< mpfr::mpreal     , Dynamic, Dynamic > matrix;
   typedef Matrix< AD<mpfr::mpreal> , Dynamic, Dynamic > a_matrix;
   //
   typedef CppAD::eigen_vector<mpfr::mpreal>          vector;
   typedef CppAD::eigen_vector< AD<mpfr::mpreal> >    a_vector;
   //

   // domain and range space vectors
   size_t size = 3, n  = size * size, m = 1;
   a_vector a_x(n), a_y(m);
   vector x(n);

   // set and declare independent variables and start tape recording
   for(size_t i = 0; i < size; i++)
   {  for(size_t j = 0; j < size; j++)
      {  // lower triangular matrix
         a_x[i * size + j] = x[i * size + j] = 0.0;
         if( j <= i )
            a_x[i * size + j] = x[i * size + j] = mpfr::mpreal(1 + i + j);
      }
   }
   CppAD::Independent(a_x);

   // copy independent variable vector to a matrix
   Index Size = Index(size);
   a_matrix a_X(Size, Size);
   // matrix     X(Size, Size);
   for(size_t i = 0; i < size; i++)
   {  for(size_t j = 0; j < size; j++)
      {  Index I = Index(i);
         Index J = Index(j);
         // X(I ,J)   = x[i * size + j];
         // If we used a_X(i, j) = X(i, j), a_X would not depend on a_x.
         a_X(I, J) = a_x[i * size + j];
      }
   }

   // Compute the log of determinant of X
   a_y[0] = log( a_X.determinant() );

   // create f: x -> y and stop tape recording
   CppAD::ADFun<mpfr::mpreal> f(a_x, a_y);
   // f.Reverse()

   // // check function value
   // mpfr::mpreal eps = 100. * CppAD::numeric_limits<mpfr::mpreal>::epsilon();
   // CppAD::det_by_minor<mpfr::mpreal> det(size);
   // ok &= NearEqual(Value(a_y[0]) , log(det(x)), eps, eps);

   // compute the derivative of y w.r.t x using CppAD
   vector jac = f.Jacobian(x);

   // check the derivative using the formula
   // d/dX log(det(X)) = transpose( inv(X) )
   // matrix inv_X = X.inverse();
   // for(size_t i = 0; i < size; i++)
   // {  for(size_t j = 0; j < size; j++)
   //    {  Index I = Index(i);
   //       Index J = Index(j);
   //       ok &= NearEqual(jac[i * size + j], inv_X(J, I), eps, eps);
   //    }
   // }

   return ok;
}

void mpfr_setup(){
    const int working_digits = 100;
    const int printing_digits = 20;
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(working_digits));
    std::cout.precision(printing_digits);
}

int main(){
    mpfr_setup();
    std::cout << std::boolalpha << eigen_det() << std::endl;
}