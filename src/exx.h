/*
 * @file exx.h
 * @brief utilities for computing exact exchange energies, including orbital and total energies.
 */
#include "matrix_m.h"
#include "meanfield.h"
#include "ri.h"
#include <set>

namespace LIBRPA
{

class Exx
{
   private:
    //! refenrence to the MeanField object to compute density matrix
    const MeanField& mf_;

    //! reference to the fractional kpoint list on which the MeanField object is computed
    const vector<Vector3_Order<double>>& kfrac_list_;

    //! period of unit cells in the BvK cell
    const Vector3_Order<int>& period_;

    bool is_rspace_build_;
    bool is_kspace_built_;
    std::set<std::string> debug_dmat_dump_tags_;
    bool debug_kstar_dumped_ = false;

    ComplexMatrix get_dmat_cplx_R_global(const int& ispin, const int& isoc1, const int& isoc2,
                                         const Vector3_Order<int>& R);
    ComplexMatrix get_dmat_cplx_R_symmetry_restored(const int& ispin, const int& isoc1,
                                                    const int& isoc2,
                                                    const Vector3_Order<int>& R);
    bool can_restore_dmat_from_abacus_symmetry() const;
    ComplexMatrix extract_dmat_cplx_R_IJblock(const ComplexMatrix& dmat_cplx, const atom_t& I,
                                              const atom_t& J);
    void maybe_dump_abacus_kstar_debug();
    void maybe_dump_full_kspace_dmat_debug(const int& ispin, const int& isoc1, const int& isoc2);
    void maybe_dump_restored_kspace_dmat_debug(const int& ispin, const int& isoc1, const int& isoc2);
    void maybe_dump_dmat_R_debug(const std::string& source_tag,
                                 const int& ispin,
                                 const int& isoc1,
                                 const int& isoc2,
                                 const Vector3_Order<int>& R,
                                 const ComplexMatrix& dmat_cplx);

    void build_dmat_R(const Vector3_Order<int>& R);
    void build_dmat_R(const atom_t& I, const atom_t& J, const Vector3_Order<int>& R);
    void warn_dmat_IJR_nonzero_imag(const ComplexMatrix& dmat_cplx, const int& ispin,
                                    const atom_t& I, const atom_t& J, const Vector3_Order<int> R);

    void build_LibRI(const Cs_LRI& Cs, const vector<Vector3_Order<int>>& Rlist,
                     const atpair_R_mat_t& coul_mat);

    void build_KS(const std::vector<std::vector<std::vector<ComplexMatrix>>>& wfc_target,
                  const std::vector<Vector3_Order<double>>& kfrac_target);

   public:
    //! Density matrix in lattice vector space, dimension (nspins, nsoc1, nsoc2, I, J, R, nao_I,
    //! nao_J)
    map<int, map<int, map<int, atpair_R_mat_t>>> dmat;

    //! exact-exchange Hamiltonian in real space, dimension (nspins, n_soc1, n_soc2, R, I, J, nao_I,
    //! nao_J)
    map<int, map<int, map<int, map<Vector3_Order<int>, map<atom_t, map<atom_t, Matd>>>>>> exx;
    map<int, map<int, map<int, map<Vector3_Order<int>, map<atom_t, map<atom_t, Matz>>>>>> exx_cplx;

    //! exact-exchange Hamiltonian in the basis of KS states, dimension (nspins, n_kpoints, n_bands,
    //! n_bands)
    map<int, map<int, Matz>> exx_is_ik_KS;

    //! exact-exchange energy of each state, dimension (nspins, n_kpoints, n_bands). This is
    //! actually the diagonal elements of Heex_KS.
    map<int, map<int, map<int, double>>> Eexx;

    Exx(const MeanField& mf, const vector<Vector3_Order<double>>& kfrac_list,
        const Vector3_Order<int>& period);

    //! Build and store the real-space exchange matrix
    template <typename Tdata>
    void build(const Cs_LRI& Cs, const vector<Vector3_Order<int>>& Rlist,
               const atpair_R_mat_t& coul_mat);

    void build_KS_kgrid();
        void build_KS_kgrid0();
    void build_KS_band(const std::vector<std::vector<std::vector<ComplexMatrix>>>& wfc_band,
                       const std::vector<Vector3_Order<double>>& kfrac_band);
    void reset_rspace();
    void reset_kspace();
};

} /* end of namespace LIBRPA */
