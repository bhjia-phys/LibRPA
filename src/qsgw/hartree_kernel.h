#pragma once

#include "../math/complexmatrix.h"
#include "../math/matrix_m.h"

#include <map>
#include <vector>

namespace librpa_int
{
namespace qsgw
{

using HartreeCkMap =
    std::map<int, std::map<int, std::map<int, ComplexMatrix>>>;
using HartreeDkMap =
    std::map<int, std::map<int, std::map<int, ComplexMatrix>>>;
using HartreeVqMap = std::map<int, std::map<int, ComplexMatrix>>;

enum class HartreeKNormalization
{
    weighted_occupations,
    legacy_extra_inverse_nk,
};

HartreeDkMap contract_hartree_full_grid(
    const HartreeCkMap& c_k,
    const HartreeVqMap& v_q0,
    const HartreeDkMap& weighted_density_k,
    const std::map<int, int>& atom_ao_sizes,
    const std::vector<int>& kpoints,
    HartreeKNormalization normalization);

} // namespace qsgw
} // namespace librpa_int
