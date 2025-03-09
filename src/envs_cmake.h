#pragma once

namespace LIBRPA
{

namespace envs
{

//! The path of the source code directory
extern const char * source_dir;

//! Git hash of the source code
extern const char * git_hash;

//! Git reference of the source code
extern const char * git_ref;

//! Cmake C++ compiler
extern const char * cxx_compiler;

//! Cmake Fortran compiler
extern const char * fortran_compiler;

//! Cmake C++ compiler flags
extern const char * cxx_compiler_flags;

//! Cmake Fortran compiler flags
extern const char * fortran_compiler_flags;

} /* end of namespace envs */
} /* end of namespace LIBRPA */
