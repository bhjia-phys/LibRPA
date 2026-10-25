# Theory notes

These notes explain the theory and numerical methods used by LibRPA. They are
read directly from the source tree and are outside the code and documentation
builds through the Sphinx `theory/**` exclusion. User and developer guides
remain in the built documentation sections.

## Shared numerical methods

- [Padé continuation and numerical breakdown](numerical_methods/pade.md)

## QSGW

- [Self-energy staticization](qsgw/staticization.md)
- [Symmetry-covariant periodic Hamiltonian images](qsgw/periodic-images.md)

## SOC and symmetry

- [Spinor response, retained subspaces and frequency symmetry](symmetry/spinor-response.md)
- [Signed-time support and self-energy covariance](symmetry/soc-self-energy-signed-time.md)
- [General magnetic response and signed screened interaction](symmetry/nonreciprocal-screening.md)
- [General-frequency strict-2D complete-Wc screening](symmetry/nonreciprocal-strict2d-screening.md)
- [RPA energy from general magnetic frequency matrices](symmetry/nonreciprocal-rpa-energy.md)
- [ABACUS writer and LibRPA reader contracts](symmetry/abacus-gw-inputs.md)
- [Atomic Bloch embedding through localized GW contractions](symmetry/bloch-embedding.md)
- [Physical input and completed GW operator lifetimes](symmetry/cache-lifetimes.md)
- [Distributed matrix ownership, empty rows/columns and scaling](symmetry/distributed-matrix-collection.md)
