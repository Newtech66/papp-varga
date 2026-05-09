- **TODO:** Use `noalias` appropriately
- ~~**TODO:** configure use of OpenMP~~ benchmarks indicate that openmp makes things worse in general. maybe the conjugate gradient could benefit from it idk
- ~~**TODO:** Use `EIGEN_NO_DEBUG` to disable assertions (?)~~ no effect in benchmarks
- **TODO:** Look into using conjugate gradient in matrix-free mode for the IHVP of `LogPerspecEpi`
- **TODO:** Right now the MPFR support is disabled. Find a way to use a CMake option to switch between
  MPFR and double build. ~~Also bring back the logperspecepi cone.~~
- **TODO:** Go further in figuring out how to bring down the compile times. Reorganize files if necessary.
- **TODO:** The current build seems to be performing slower than the original non-CMake build. I don't know why.
  The difference is 312 seconds for the new build vs 198 seconds for the old build. Which is just absurd.
  This is despite compiling with OpenMP and -DNDEBUG.
- **TODO:** Using the realView API (or in other words changing the output order of real and imaginary parts) breaks the solver.
  I don't know why.
