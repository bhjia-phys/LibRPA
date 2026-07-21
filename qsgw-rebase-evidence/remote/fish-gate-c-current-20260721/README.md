# Fish Gate C: current QSGW Hartree

This gate runs the frozen Si k444 symmetry-on ABACUS dataset with the current
contract-v6 executable for two completed updates in both `qsgw` and
`qsgw_band` tasks. Both runs use linear mixing with beta 0.2, full Coulomb,
weighted occupations, and disabled head-wing updates.

The gate validates:

- exact immutable bundle, contract, executable, and Gate 0 provenance;
- iteration-1 zero Hartree delta and iteration-2 nonzero response;
- density charge/Hermiticity, exact full-k/R ordering, and atom-pair BvK remap;
- independent `C(k)-V(q=0)-density` contraction from the frozen `Cs_data_*`
  and full-Coulomb producer files for both task paths;
- independent inverse Fourier reconstruction and fixed-basis projection onto
  both SCF-grid and band channels;
- current-v6 raw Hamiltonian, mixing, diagonalization, band-table, and CSR
  closure through the Gate D validator; and
- equality of the SCF-grid channel between `qsgw` and `qsgw_band` runs.

This is a current-code structural and metamorphic gate. It does not replace the
separate same-dataset legacy/current numerical Hartree acceptance run.
