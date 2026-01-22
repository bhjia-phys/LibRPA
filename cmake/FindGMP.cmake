# FindGMP.cmake
# Try to find GMP library

find_path(GMP_INCLUDE_DIR
  NAMES gmp.h
  PATHS
    /usr/include
    /usr/local/include
    /usr/include/x86_64-linux-gnu
    ENV GMP_ROOT
    ENV GMP_DIR
  PATH_SUFFIXES
    include
)

find_library(GMP_LIBRARY
  NAMES gmp libgmp
  PATHS
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    ENV GMP_ROOT
    ENV GMP_DIR
  PATH_SUFFIXES
    lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARY GMP_INCLUDE_DIR)

if(GMP_FOUND)
  set(GMP_LIBRARIES ${GMP_LIBRARY})
  set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
endif()
