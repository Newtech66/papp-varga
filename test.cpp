#include <iostream>
#include <complex>
#include <type_traits>

int main(){
    std::complex<double> a;
    if(std::is_same<decltype(a)::value_type, double>::value){
        std::cout << "YES" << std::endl;
    }
}