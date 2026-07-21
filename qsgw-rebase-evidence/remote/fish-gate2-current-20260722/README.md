# Fish Gate2: current QSGW first self-energy

This gate uses the accepted symmetry-on Si k444 Gate1 input and upstream G0W0
SigmaC blocks. It runs the current candidate with `task=qsgw`, no mixing, and
exactly one completed update.

The iteration labels are explicit:

- trace iteration 0 is the immutable producer/KS initial state;
- trace iteration 1, channel 0 is the first self-energy evaluated from that
  initial state;
- that first QSGW SigmaC is compared with the accepted upstream G0W0 SigmaC.

The v6 observers consume the unmodified `qsgw_contract_version 6` files. They
validate the actual symmetry/head-wing/Hartree/band/mixer contract, the input
contract hash, initial occupations/eigenvalues/Fermi level, charge, full
Hamiltonian closure, no-mixer identity, Hermiticity, rotation unitarity,
diagonalization, fixed-basis wavefunction rotation, and all 48 SigmaC blocks.

The runner binds the accepted Gate0 and Gate1 archives, candidate executable,
frozen ABACUS bundle, physical `vxc_out`, dimensionless pinned-producer
symmetry tail, structure overlay builder, derived-contract builder, formal
regression parser, and all observers by SHA256. The derived contract changes
only the `reader_static ... stru_out` SHA so the strict QSGW preflight binds
the composite structure; every other contract byte remains unchanged. A
successful run writes `GREEN_CONFIRMED`; any failure writes `FAILED`.

This gate does not establish multi-round legacy/current parity, Hartree,
`qsgw_band`, or H_QSGW cut correctness. Those remain separate gates.
