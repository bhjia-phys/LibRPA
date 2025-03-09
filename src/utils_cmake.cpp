#include "utils_cmake.h"

#include "envs_cmake.h"
#include "utils_io.h"

namespace LIBRPA
{

namespace utils
{

void print_cmake_info()
{
    lib_printf("LibRPA CMake build info:\n");
    lib_printf("| Source code directory    : %s\n", envs::source_dir);
    lib_printf("| Source code Git reference: %s\n", envs::git_ref);
    lib_printf("| Source code Git hash     : %s\n", envs::git_hash);
    lib_printf("| C++ compiler             : %s\n", envs::cxx_compiler);
    lib_printf("| C++ compiler flags       : %s\n", envs::cxx_compiler_flags);
    lib_printf("| Fortran compiler         : %s\n", envs::fortran_compiler);
    lib_printf("| Fortran compiler flags   : %s\n", envs::fortran_compiler_flags);
}

} /* end of namespace utils */

} /* end of namespace LIBRPA */
