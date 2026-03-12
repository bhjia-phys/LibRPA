/*!
 * @file abacus_symmetry.h
 * @brief Utilities for reading ABACUS symmetry sidecar files.
 */
#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "atoms.h"
#include "complexmatrix.h"
#include "vector3_order.h"

namespace LIBRPA
{

using abacus_R_t = std::array<int, 3>;
using abacus_irreducible_sector_t = std::map<atpair_t, std::set<abacus_R_t>>;

/*!
 * @brief Real-space symmetry operation exported by ABACUS.
 */
struct AbacusSymmetryOperation
{
    int isym = -1;
    std::array<std::array<double, 3>, 3> rotation{{{{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}},
                                                    {{0.0, 0.0, 0.0}}}};
    Vector3_Order<double> translation{0.0, 0.0, 0.0};
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief Atom-resolved k-space symmetry information exported by ABACUS.
 */
struct AbacusKAtomRotation
{
    int atom_from = -1;
    int atom_to = -1;
    int atom_type = -1;
    int lmax = -1;
    std::map<int, ComplexMatrix> shell_rotations;
};

/*!
 * @brief One member of an irreducible k-star.
 */
struct AbacusKStarMember
{
    int isym = -1;
    Vector3_Order<double> k_bz{0.0, 0.0, 0.0};
    std::vector<AbacusKAtomRotation> atom_rotations;
};

/*!
 * @brief One irreducible k-star exported by ABACUS.
 */
struct AbacusKStar
{
    int star_index = -1;
    Vector3_Order<double> k_ibz{0.0, 0.0, 0.0};
    std::vector<AbacusKStarMember> members;
};

/*!
 * @brief AO shell layout of one ABACUS atom type.
 *
 * The shell multiplicities follow the ABACUS orbital ordering:
 * increasing angular momentum `l`, then zeta index, then magnetic index.
 */
struct AbacusAOTypeLayout
{
    std::string label;
    std::string orbital_file;
    std::vector<int> shell_counts;
    int nao = 0;
};

/*!
 * @brief One full real-space member generated from an irreducible {atom pair, R}.
 */
struct AbacusRSpaceRestoreMember
{
    int isym = -1;
    atpair_t full_atom_pair;
    Vector3_Order<int> full_R{0, 0, 0};
};

using abacus_rspace_sector_stars_t =
    std::map<atpair_t, std::map<Vector3_Order<int>, std::vector<AbacusRSpaceRestoreMember>>>;

/*!
 * @brief In-memory representation of ABACUS symmetry sidecar files.
 *
 * The context is intentionally read-only after loading. It will be used by later
 * EXX/GW symmetry implementations to avoid reparsing the sidecar files.
 */
struct AbacusSymmetryContext
{
    bool available = false;
    bool ao_shell_layout_available = false;
    int ao_lmax = -1;
    int abf_lmax = -1;
    abacus_irreducible_sector_t irreducible_sector;
    std::vector<AbacusSymmetryOperation> rspace_operations;
    std::vector<AbacusKStar> kstars;
    std::vector<AbacusAOTypeLayout> ao_type_layouts;
    std::map<atom_t, int> atom_to_type;

    void clear();
    bool empty() const;
    bool has_ao_shell_layout() const;
    std::size_t count_irreducible_pairs() const;
    std::size_t count_irreducible_blocks() const;
    std::size_t count_kstar_members() const;
    std::size_t count_atoms_with_layout() const;
    const AbacusAOTypeLayout& get_ao_type_layout(int atom_type) const;
};

extern AbacusSymmetryContext abacus_symmetry_ctx;

bool load_abacus_symmetry_context(const std::string& dir_path,
                                  AbacusSymmetryContext& ctx,
                                  std::ostream* log = nullptr);

bool load_global_abacus_symmetry_context(const std::string& dir_path,
                                         std::ostream* log = nullptr);

ComplexMatrix build_abacus_ao_rotation_matrix(const AbacusSymmetryContext& ctx,
                                              int atom_type,
                                              const std::map<int, ComplexMatrix>& shell_rotations);

ComplexMatrix rotate_abacus_kspace_matrix(const AbacusSymmetryContext& ctx,
                                          const AbacusKStarMember& member,
                                          const ComplexMatrix& matrix_ibz,
                                          const std::map<atom_t, size_t>& atom_nw,
                                          const Vector3_Order<double>& k_ibz,
                                          const std::map<atom_t, std::array<double, 3>>& coord_frac,
                                          bool use_time_reversal = false);

void build_abacus_rspace_sector_stars(
    const AbacusSymmetryContext& ctx,
    const std::map<atom_t, std::array<double, 3>>& coord_frac,
    const Vector3_Order<int>& period,
    const std::vector<Vector3_Order<int>>& Rlist,
    abacus_rspace_sector_stars_t& sector_stars,
    std::ostream* log = nullptr);

ComplexMatrix rotate_abacus_rspace_matrix(const AbacusSymmetryContext& ctx,
                                          int isym,
                                          atom_t atom_from_i,
                                          atom_t atom_from_j,
                                          const ComplexMatrix& matrix_source);

} // namespace LIBRPA
