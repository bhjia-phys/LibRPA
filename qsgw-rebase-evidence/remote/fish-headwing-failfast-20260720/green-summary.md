# QSGW head-wing fail-fast GREEN gate

- Candidate: `44059ebf4dbe2c76f2bfa7598baa388e14b0205e`
- Upstream base: `42d3863c1d865194d382a085851d1e2e8a39764f`
- Host: `Fisherd-Server100.96.1.64`
- Passing packet: `green-v3/`
- Executable SHA256: `14343853a0e9857d90426bbe40cd1f2dec5cc735c152c3b67529f7b1bd4e391d`

The focused `test_qsgw_inputfile` gate passed and checks both `qsgw` and
`qsgw_band` fail-fast behavior while retaining a G0W0 analytic head-wing
parser control. The oneAPI build then completed and CTest passed 60/60 with
zero failed and zero Not Run tests. `test_rpa_headwing` remained passing.
The protected diff against the upstream base is empty for `src/core`,
`driver/tasks/g0w0.cpp`, and `driver/tasks/rpa.cpp`.

`green-v1/` is a retained setup failure: LibRPA's Git revision CMake helper
cannot configure a linked worktree created from the bare repository.
`green-v2/` is a retained toolchain failure: the focused gate and the driver
compiled, but a default GNU C++ configuration could not link Intel threaded
MKL MPI tests because the Intel OpenMP runtime was absent from the link.
`green-v3/` uses the same `mpiicpx`/`mpiifx` oneAPI configuration as the
previous full-suite gate and is the acceptance result.
