#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <complex>
#include <map>
#include <stdexcept>
#include <vector>

#include "atomic_basis.h"
#include "symmetry_context.h"
#include "../math/complexmatrix.h"
#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#include "../utils/constants.h"

namespace librpa_int
{

// Atomic embedding at a fixed Bloch-sum sign. Raw coefficients obey
// C_eta(k)=B(k) C_cell(k), B_I=exp[-2pi i s eta k.tau_I]. The localized
// RI tensors retain their integer-cell frame. Apply only to existing work
// matrices, leaving the input coefficients and symmetry context unchanged.
// This does not convert the independent Bloch-sign/Fourier convention.
class BlochEmbedding
{
    double coefficient_ = 0.;
    std::vector<std::array<double, 3>> positions_;
    std::vector<atom_t> ao_atoms_;

  public:
    BlochEmbedding() = default;

    BlochEmbedding(const SymmetryContext &context, const AtomicBasis &basis)
        : BlochEmbedding(context, basis.get_atom_nb_map<atom_t>())
    {
    }

    BlochEmbedding(const SymmetryContext &context, const std::map<atom_t, size_t> &atom_nw)
    {
        const auto &convention = context.basis_convention;
        if (convention.bloch_ratom == 0 || convention.bloch_ratom == LIBRPA_UNSET)
            return;
        if ((convention.bloch_phase != -1 && convention.bloch_phase != 1)
            || (convention.bloch_ratom != -1 && convention.bloch_ratom != 1))
            throw std::invalid_argument("Atomic Bloch embedding requires a declared Bloch convention");
        if (atom_nw.empty())
            throw std::invalid_argument("Atomic Bloch embedding requires an AO/auxiliary atom partition");

        coefficient_ = convention.bloch_phase * convention.bloch_ratom;
        positions_.resize(static_cast<size_t>(atom_nw.rbegin()->first) + 1);
        for (const auto &[atom, count] : atom_nw)
        {
            if (count == 0)
                continue;
            const auto coordinate = context.input_coord_frac.find(atom);
            if (coordinate == context.input_coord_frac.end())
                throw std::invalid_argument("Atomic Bloch embedding requires fractional atomic coordinates");
            for (const auto value : coordinate->second)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Atomic Bloch embedding requires finite atomic coordinates");
            positions_[atom] = coordinate->second;
            ao_atoms_.insert(ao_atoms_.end(), count, atom);
        }
    }

    bool active() const { return coefficient_ != 0.; }

    // Empty means the exact identity path: no phases, allocations or matrix
    // multiplications are needed for existing cell-periodic inputs.
    std::vector<std::complex<double>> atom_phases(const Vector3_Order<double> &kfrac) const
    {
        if (!active())
            return {};
        std::vector<std::complex<double>> phases(positions_.size());
        for (size_t atom = 0; atom != positions_.size(); ++atom)
        {
            const auto &tau = positions_[atom];
            const double angle = -TWO_PI * coefficient_
                                 * (kfrac.x * tau[0] + kfrac.y * tau[1] + kfrac.z * tau[2]);
            phases[atom] = {std::cos(angle), std::sin(angle)};
        }
        return phases;
    }

    void apply_bilinear(ComplexMatrix &matrix, const Vector3_Order<double> &kfrac,
                        bool to_cell = true) const
    {
        if (!active())
            return;
        if (matrix.nr != static_cast<int>(ao_atoms_.size()) || matrix.nc != matrix.nr)
            throw std::invalid_argument("Atomic Bloch bilinear has an inconsistent basis dimension");
        const auto phases = atom_phases(kfrac);
        for (int row = 0; row != matrix.nr; ++row)
            for (int col = 0; col != matrix.nc; ++col)
            {
                auto phase = std::conj(phases[ao_atoms_[row]]) * phases[ao_atoms_[col]];
                if (!to_cell)
                    phase = std::conj(phase);
                matrix(row, col) *= phase;
            }
    }

    void apply_bilinear(Matz &matrix, const ArrayDesc &descriptor,
                        const Vector3_Order<double> &kfrac, bool to_cell = true) const
    {
        if (!active())
            return;
        if (descriptor.m() != static_cast<int>(ao_atoms_.size())
            || descriptor.n() != descriptor.m() || matrix.nr() != descriptor.m_loc()
            || matrix.nc() != descriptor.n_loc())
            throw std::invalid_argument("Distributed atomic Bloch bilinear has an inconsistent basis layout");
        const auto phases = atom_phases(kfrac);
        for (int row = 0; row != descriptor.m_loc(); ++row)
            for (int col = 0; col != descriptor.n_loc(); ++col)
            {
                const auto i = descriptor.indx_l2g_r(row);
                const auto j = descriptor.indx_l2g_c(col);
                auto phase = std::conj(phases[ao_atoms_[i]]) * phases[ao_atoms_[j]];
                if (!to_cell)
                    phase = std::conj(phase);
                matrix(row, col) *= phase;
            }
    }
};

} // namespace librpa_int
