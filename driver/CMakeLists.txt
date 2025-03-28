add_executable(rpa_exe)
set_target_properties(rpa_exe PROPERTIES
  OUTPUT_NAME ${DRIVER_NAME}
  ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND exe_sources
  driver_params.cpp
  driver_utils.cpp
  main.cpp
  read_data.cpp
  write_aims.cpp
  inputfile.cpp
  task_rpa.cpp
  task_exx.cpp
  task_gw.cpp
  task_gw_band.cpp
  task_qsgw.cpp
  task_qsgwA.cpp
  task_qsgw_band.cpp
  task_hf_band.cpp
  task_scRPA.cpp
  task_scRPA_band.cpp
  task_screened_coulomb.cpp
  )

target_include_directories(rpa_exe
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/qsgw
  )

target_sources(rpa_exe PRIVATE ${exe_sources})
if(USE_LIBRI)
  target_include_directories(rpa_exe
    PRIVATE
      ${Cereal_INCLUDE_DIR}
      ${LibComm_INCLUDE_DIR}
      ${LibRI_INCLUDE_DIR}
    )
endif()

target_link_libraries(rpa_exe PRIVATE rpa_lib qsgw_lib ${math_libs} ${parallel_libs})

install(TARGETS rpa_exe DESTINATION ${CMAKE_INSTALL_BINDIR})
