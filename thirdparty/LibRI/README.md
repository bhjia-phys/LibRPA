# LibRI

Read this in other languages: [English](./README.md), [中文](./README.cn.md)


## RI technique

Conventional Density functional theory (DFT) methods are not sufficient to properly describe certain types of binding interactions.
It is often necessary to use higher-level methods like hybrid functionals, random-phase approximation (RPA), GW methods and beyond to accurately treat.
However, the computational cost of these methods are prohibitively expensive.

For local bases, invoking the resolution-of-the-identity (RI) significantly reduces the storage requirement and the computational cost.


## requirements for LibRI

LibRI is a C++ library using RI technique for methods beyond DFT. It is a head-only library.

These are necessary for LibRI:
- C++ compiler, supported C++14 and OpenMP.
- MPI library, used for inter-process communication.
- BLAS and LAPACK libraries, used as backends for Tensor operations.
  > If Math Kernel Library (MKL) is linked as the BLAS and LAPACK backend, it's recommended to define macro `__MKL_RI` before including any LibRI's header file. Some LibRI's functions will be silently substituted with calls to MKL routines.
- [cereal](https://uscilab.github.io/cereal/) library, a head-only library for serialization.
- [LibComm](https://github.com/abacusmodeling/LibComm.git) library, a head-only library for inter-process communication.

## LibRPA Hartree compatibility subset

This bundled tree retains its `75902f8` base and existing LibRPA patches.
The Hartree/k-space headers and their required helper additions are backported
from official LibRI `2306483f539b7d111363c48d85b6491dd9f226bd`, with the auxiliary-potential
interface from `a19260ad726be4b922897948d12a284e8afa26ff`. This is a scoped
compatibility subset, not a replacement by that complete upstream version.

The existing EXX/LRI contraction kernels and Tensor storage remain unchanged.
`Tensor::conjugate`, terminal-value `find`, `set_union`, spin-independent k
schedulers and additional Hartree accumulation/lock helper overloads provide
the interfaces used by the new headers. Existing helper definitions are retained.
The new tensor GEMV output constructors use this tree's fixed-size
`Shape_Vector`; the original shared `valarray` representation is preserved.
No local-interaction/HSE headers are included in this subset.

`RI_HARTREE_HAS_AUXILIARY_POTENTIAL` identifies the optional auxiliary-potential
API. `unittests/physics/hartree_auxiliary_standalone.cpp` checks the communicated
potential and the unchanged ordinary Hartree result against a closed-form
one-atom/two-k-point reference at one and two MPI ranks.
