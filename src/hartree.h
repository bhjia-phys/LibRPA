/*
 * @file Hartree.h
 * @brief utilities for computing Hartree energies, including orbital and total energies.
 * @author Ziqing Guan
 */
#include "matrix_m.h"
#include "meanfield.h"
#include "ri.h"

namespace LIBRPA
{

class Hartree
{
   private:
    //! refenrence to the MeanField object to compute density matrix
    const MeanField& mf_;
    Vector3_Order<int> period_; 
    //! reference to the fractional kpoint list on which the MeanField object is computed
    const vector<Vector3_Order<double>>& kfrac_list_;
    vector<std::array<double, 3>> kfrac_array_list_;
    
    // map<int, map<int, map<int, map<Vector3_Order<int>, map<atom_t, map<atom_t, Matd>>>>>> hartree;
    
    //! period of unit cells in the BvK cell
    bool is_rspace_build_;
    bool is_kspace_built_;

    ComplexMatrix get_dmat_cplx_k_global(const int ik);
    ComplexMatrix extract_dmat_cplx_IJblock(const ComplexMatrix& dmat_cplx, const atom_t& I,
                                              const atom_t& J);

    void warn_IJR_nonzero_imag(const RI::Tensor<std::complex<double>> &t,
                                const atom_t I, const atom_t J, const std::array<int, 3>& R);

    std::array<int, 3> nearest_R(int I, int J, const std::array<int, 3> &R);

    void build_KS(const std::vector<std::vector<std::vector<ComplexMatrix>>>& wfc_target,
                  const std::vector<Vector3_Order<double>>& kfrac_target);

   public:
    // ! Hartree Hamiltonian in real space, dimension (I, J, R, nao_I, nao_J), spin and soc is summed
    map<int, map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> HHartree_libri;

    //! Hartree Hamiltonian in the basis of KS states, dimension (nspins, n_kpoints, n_bands, n_bands)
    map<int, map<int, Matz>> Hartree_is_ik_KS;
    map<int, map<int, Matz>> Hartree_is_ik_nao;
    //! Hartree energy of each state, dimension (nspins, n_kpoints, n_bands). This is
    //! actually the diagonal elements of Heex_KS.
    map<int, map<int, map<int, double>>> EHartree;

    Hartree(const MeanField& mf, const vector<Vector3_Order<double>>& kfrac_list,
        const Vector3_Order<int>& period);

    //! Build and store the real-space Hartree matrix
    void build(const Cs_LRI& Cs, const vector<Vector3_Order<int>>& Rlist,
               const atpair_R_mat_t& coul_mat);

    void build_KS_kgrid();
    void build_KS_kgrid0();
    void build_KS_band(const std::vector<std::vector<std::vector<ComplexMatrix>>>& wfc_band,
                       const std::vector<Vector3_Order<double>>& kfrac_band);
    void reset_rspace();
    void reset_kspace();
    
    inline void print_a(std::ostream& ofs, const std::vector<int>& vec, const std::string name)
    {
        ofs << name << ": ";
        int count = 0;
        for (auto& v : vec)
        {
            ofs << v << " ";
            count++;
            if (count % 10 == 0) { ofs << std::endl; }
        }
        ofs << std::endl;
    }
};

} /* end of namespace LIBRPA */
