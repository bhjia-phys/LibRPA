#include "hartree.h"
#include "constants.h"
#include "envs_blacs.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "geometry.h"
#include "lapack_connector.h"
#include "libri_utils.h"
#include "matrix_m_parallel_utils.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "stl_io_helper.h"
#include "utils_blacs.h"
#include "vector3_order.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/physics/Hartree.h>
#include <RI/ri/Cell_Nearest.h>
#else
#include "libri_stub.h"
#endif
#include "utils_io.h"

namespace LIBRPA
{

Hartree::Hartree(const MeanField &mf, const vector<Vector3_Order<double>> &kfrac_list,
         const Vector3_Order<int> &period)
    : mf_(mf), kfrac_list_(kfrac_list), period_(period)
{
    is_rspace_build_ = false;
    is_kspace_built_ = false;
    for (auto &k: kfrac_list_)
    {
        this->kfrac_array_list_.emplace_back(std::array<double,3>{k.x, k.y, k.z});
    }
};

ComplexMatrix Hartree::get_dmat_cplx_k_global(const int ik)
{
    // summation rho's spin channel
    const auto nspins = this->mf_.get_n_spins();
    const auto nsoc = this->mf_.get_n_soc();
    ComplexMatrix dmat_cplx(this->mf_.get_n_aos(), this->mf_.get_n_aos());
    for (int ispin = 0; ispin < nspins; ispin++)
    {
        for (int isoc = 0; isoc < nsoc; isoc++)
        {
            dmat_cplx += this->mf_.get_dmat_cplx(ispin, isoc, isoc, ik);                
        }
    }
    return dmat_cplx;
}

ComplexMatrix Hartree::extract_dmat_cplx_IJblock(const ComplexMatrix &dmat_cplx, const atom_t &I, const atom_t &J)
{
    const auto I_num = atom_nw.at(I);
    const auto J_num = atom_nw.at(J);
    ComplexMatrix dmat_cplx_IJ(I_num, J_num);
    for (size_t i = 0; i != I_num; i++)
    {
        size_t i_glo = atom_iw_loc2glo(I, i);
        for (size_t j = 0; j != J_num; j++)
        {
            size_t j_glo = atom_iw_loc2glo(J, j);
            dmat_cplx_IJ(i, j) = dmat_cplx(i_glo, j_glo);
        }
    }
    return dmat_cplx_IJ;
}

void Hartree::warn_IJR_nonzero_imag(const RI::Tensor<std::complex<double>> &t,
                                     const atom_t I, const atom_t J, const std::array<int, 3> &R)
{
    for (size_t i = 0; i < t.get_shape_all(); ++i)
    {
        if (std::abs(std::imag((*t.data)[i])) > 1e-2)
        {
            utils::lib_printf(
                "Warning: complex-valued density matrix, IJR %zu %zu (%d, %d, %d)\n", I,
                J, R[0], R[1], R[2]);
            break;
        }
    }
}

std::array<int, 3> Hartree::nearest_R(int I, int J, const std::array<int, 3> &R){
    auto distsq = std::numeric_limits<double>::max();
    Vector3<int> R_IJ;
    std::array<int, 3> R_bvk;
    for (int i = -1; i < 2; i++)
    {
        R_IJ.x = i * this->period_.x + R[0];
        for (int j = -1; j < 2; j++)
        {
            R_IJ.y = j * this->period_.y + R[1];
            for (int k = -1; k < 2; k++)
            {
                R_IJ.z = k * this->period_.z + R[2];
                const auto diff =
                    (Vector3<double>(coord_frac[I][0], coord_frac[I][1], coord_frac[I][2]) -
                        Vector3<double>(coord_frac[J][0], coord_frac[J][1], coord_frac[J][2]) -
                        Vector3<double>(R_IJ.x, R_IJ.y, R_IJ.z)) *
                    latvec;
                const auto norm2 = diff.norm2();
                if (norm2 < distsq)
                {
                    distsq = norm2;
                    R_bvk[0] = R_IJ.x;
                    R_bvk[1] = R_IJ.y;
                    R_bvk[2] = R_IJ.z;
                }
            }
        }
    }
    return R_bvk;
}


void Hartree::build(const Cs_LRI &Cs, const vector<Vector3_Order<int>> &Rlist,
                const atpair_R_mat_t &coul_mat)
{
    using LIBRPA::envs::mpi_comm_global;
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::atomic_basis_abf;
    using LIBRPA::atomic_basis_wfc;
    assert(parallel_routing == ParallelRouting::LIBRI);

    if (this->is_rspace_build_)
    {
        return;
    }

    const int n_spins = this->mf_.get_n_spins();
    const int n_soc = this->mf_.get_n_soc();
    const int nk = this->mf_.get_n_kpoints();
    const int natom = atom_mu.size();

#ifdef LIBRPA_USE_LIBRI
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf("Computing Hartree orbital energy using LibRI\n");
    }
    mpi_comm_global_h.barrier();

    RI::Hartree<int, int, 3, std::complex<double>> hartree_libri;
    map<int, std::array<double, 3>> atoms_pos;
    for (int i = 0; i != atom_mu.size(); i++)
        atoms_pos.insert(pair<int, std::array<double, 3>>{i, {0, 0, 0}});

    std::array<double, 3> xa{latvec.e11, latvec.e12, latvec.e13};
    std::array<double, 3> ya{latvec.e21, latvec.e22, latvec.e23};
    std::array<double, 3> za{latvec.e31, latvec.e32, latvec.e33};
    std::array<std::array<double, 3>, 3> lat_array{xa, ya, za};
    std::array<int, 3> period_array{period_.x, period_.y, period_.z};
    hartree_libri.set_parallel(mpi_comm_global, atoms_pos, lat_array, period_array);

    std::vector<int> list_I, list_J, list_IJ, list_k_index;
    std::set<int> set_I, set_J, set_IJ;
    
    int task_sizes = natom * natom * nk;
    std::cout << "Distribute Hartree task: "<< task_sizes << "Number of atom: " << natom 
        << "Number of nk:" << nk << std::endl;
    // get local IJk task list
    RI::Distribute_Equally::distribute_atom_pair_and_k(mpi_comm_global, natom, nk, list_I, list_J,
                                                       list_k_index, false);
    this->print_a(envs::ofs_myid, list_I, "Hartree list_I");
    this->print_a(envs::ofs_myid, list_J, "Hartree list_J");
    this->print_a(envs::ofs_myid, list_k_index, "Hartree list_k_index");
    
    set_I.insert(list_I.begin(), list_I.end());
    set_J.insert(list_J.begin(), list_J.end());
    set_IJ.insert(set_I.begin(), set_I.end());
    set_IJ.insert(set_J.begin(), set_J.end());
    list_IJ.assign(set_IJ.begin(), set_IJ.end());
    // for(int i = 0; i != natom; i++)
    // {
    //     set_all_atom.insert(i);
    //     list_all_atom.push_back(i);
    // }

    Profiler::start("build_real_space_Hartree_1", "Prepare C libRI object");
    envs::ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << "\n";
    // print_keys(envs::ofs_myid, Cs.data_libri);

    // convert Cs<double> to Cs<complex<double>>, since hartree_libri requires complex type
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> C_libri_cplx;
    for (const auto &I_JR_C : Cs.data_libri)
    {
        const auto I = I_JR_C.first;
        for (const auto &JR_C : I_JR_C.second)
        {
            const auto J = JR_C.first.first;
            const auto R = JR_C.first.second;
            const auto &C = JR_C.second;
            auto JR = std::pair<int, std::array<int, 3>>(J, R);
            C_libri_cplx[I][JR] = RI::Global_Func::convert<std::complex<double>>(C);
        }
    }
    hartree_libri.set_Cs(C_libri_cplx, Params::libri_exx_threshold_C, set_IJ, set_IJ);
   
    Profiler::stop("build_real_space_Hartree_1");
    envs::ofs_myid << "Finished setup Cs for Hartree\n";
    std::flush(envs::ofs_myid);

    // initialize Coulomb matrix
    Profiler::start("build_real_space_Hartree_2", "Prepare V libRI object");
    RI::Tensor<double> V_double_tmp;
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> V_libri_cplx;
    Profiler::start("build_real_space_Hartree_2_1");

    for (const auto &I_JRV : coul_mat)
    {
        const auto I = I_JRV.first;
        for (const auto &J_RV : I_JRV.second)
        {
            const auto J = J_RV.first;
            for (const auto &R_V : J_RV.second)
            {
                const auto &R = R_V.first;
                const auto &V = R_V.second;
                std::array<int, 3> Ra{R.x, R.y, R.z};
                std::valarray<double> VIJR_va;
                // utils::lib_printf("Checking  R1: (%d,%d,%d)\n", 
                //              R.x, R.y, R.z);
                VIJR_va = std::valarray<double>(V->c, V->size);
                auto pv = std::make_shared<std::valarray<double>>();
                *pv = VIJR_va;
                V_double_tmp = RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                V_libri_cplx[I][{J, Ra}] = RI::Global_Func::convert<std::complex<double>>(V_double_tmp);
            }
        }
    }

    Profiler::cease("build_real_space_Hartree_2_1");
    envs::ofs_myid << "Number of V keys: " << get_num_keys(V_libri_cplx) << "\n";
    Profiler::start("build_real_space_Hartree_2_2");
    hartree_libri.set_Vs(V_libri_cplx, Params::libri_exx_threshold_V, set_I, set_IJ);
    V_libri_cplx.clear();
    Profiler::cease("build_real_space_Hartree_2_2");
    Profiler::cease("build_real_space_Hartree_2");
    utils::lib_printf("Task %4d: V setup for Hartree\n", mpi_comm_global_h.myid);
    
    Profiler::start("build_real_space_Hartree_3", "Prepare DM libRI object");
    std::map<int, std::map<int, std::map<int, RI::Tensor<std::complex<double>>>>> dmat_libri;
    // <I,<J,<k, Tensor>>>
    for (const int k : list_k_index)
    {
        const auto dmat_cplx = this->get_dmat_cplx_k_global(k);

        for (const int I : list_I)
        {
            for (const int J: list_J)
            {
                const auto dmat_IJk =
                    this->extract_dmat_cplx_IJblock(dmat_cplx, I, J);
                std::valarray<std::complex<double>> dmat_va;
                
                dmat_va = std::valarray<std::complex<double>>(dmat_IJk.c, dmat_IJk.size);
                auto pdmat = std::make_shared<std::valarray<std::complex<double>>>();
                *pdmat = dmat_va;
                dmat_libri[I][J][k] = RI::Tensor<std::complex<double>>(
                    {size_t(dmat_IJk.nr), size_t(dmat_IJk.nc)}, pdmat);
            }
        }
    }
    Profiler::stop("build_real_space_Hartree_3");
    utils::lib_printf("Task %4d: DM setup for Hartree\n", mpi_comm_global_h.myid);

    Profiler::start("build_real_space_Hartree_4", "Calculate Hartree");

    Profiler::start("build_real_space_Hartree_4_1", "Call libRI HHartree(<I,J,k>)");
    std::map<int, std::map<int, std::map<int, RI::Tensor<std::complex<double>>>>> HHartree_k =
        hartree_libri.lri.cal_cvcd_k_hartree(dmat_libri, this->kfrac_array_list_,
                                            list_k_index, list_I, list_J, list_IJ);
    Profiler::stop("build_real_space_Hartree_4_1");
    utils::lib_printf("Task %4d: cal_cvcd_k_hartree elapsed time: %f\n",
                        mpi_comm_global_h.myid,
                        Profiler::get_wall_time_last("build_real_space_Hartree_4_1"));
    envs::ofs_myid << "Number of HHartree (<I,<J,<k,Tensor>>>) keys: "
                    << get_num_keys(HHartree_k) << "\n";

    Profiler::start("build_real_space_Hartree_4_2",
        "Convert HHartree (<I,<J,<k,Tensor>>>) to nearest (<I,<<J,R>,Tensor>>)");

    for (const auto &I_HartreeJk : HHartree_k)
    {
        const int I = I_HartreeJk.first;
        for (const auto &J_Hartreek : I_HartreeJk.second)
        {
            const int J = J_Hartreek.first;
            for (auto &R : Rlist)
            {
                const auto R_bvk = this->nearest_R(I, J, {R.x, R.y, R.z});
                RI::Tensor<std::complex<double>> HIJR({atom_nw.at(I), atom_nw.at(J)});
                for (const auto &k_Hartree : J_Hartreek.second)
                {
                    const int k = k_Hartree.first;
                    const auto &Hartree_k_tensor = k_Hartree.second;
                    
                    double phase_angle = -TWO_PI * (kfrac_list_[k].x * R_bvk[0] +
                                                    kfrac_list_[k].y * R_bvk[1] +
                                                    kfrac_list_[k].z * R_bvk[2]);
                    std::complex<double> phase_factor{std::cos(phase_angle),
                                                        std::sin(phase_angle)};
                    HIJR += Hartree_k_tensor * (phase_factor / double(nk));
                }
                this->HHartree_libri[I][{J, R_bvk}] = std::move(HIJR);
            }
        }
    }
    Profiler::stop("build_real_space_Hartree_4_2");
    envs::ofs_myid << "Number of HHartree_libri (<I,<<J,R>,Tensor>>) keys: "
                    << get_num_keys(this->HHartree_libri) << "\n";
    Profiler::stop("build_real_space_Hartree_4");
    // NOTE: For atom pair <I,J>, each process have all R, but they come from part of k-points
    // Fourier Transformation. So in the step 5, we will do `comm_map2_first` to gather complete
    // R which comes from all k-points.


#else
    if (mpi_comm_global_h.is_root())
    {
        utils::lib_printf(
            "Error: trying build Hartree orbital energy with LibRI, but the program is not compiled "
            "against LibRI\n");
    }
    throw std::logic_error("compilation");
    mpi_comm_global_h.barrier();
#endif

    is_rspace_build_ = true;
}



void Hartree::build_KS(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_target,
                   const std::vector<Vector3_Order<double>> &kfrac_target)
{
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

    assert(this->is_rspace_build_);
    // Reset k-space matrices built from last call
    if (this->is_kspace_built_)
    {
        utils::lib_printf("Warning: reset Hartree k-space matrices\n");
        this->reset_kspace();
    }

    const auto &n_aos = this->mf_.get_n_aos();
    const auto &n_spins = this->mf_.get_n_spins();
    const auto &n_bands = this->mf_.get_n_bands();
    const auto &n_soc = this->mf_.get_n_soc();

    // prepare scalapack array descriptors
    Array_Desc desc_nao_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nao(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband(blacs_ctxt_global_h);
    Array_Desc desc_nband_nband_fb(blacs_ctxt_global_h);

    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);

    // local 2D-block submatrices
    auto HHartree_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<complex<double>>(desc_nband_nao, MAJOR::COL);
    auto HHartree_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);
    auto HHartree_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);

    const auto set_IJ_naonao = LIBRPA::utils::get_necessary_IJ_from_block_2D(
        atomic_basis_wfc, atomic_basis_wfc, desc_nao_nao);
    const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_naonao);

    // collect necessary data
    Profiler::start("build_real_space_Hartree_5", "Collect HHartree IJ from world and check real");

    // Collect the IJ pair of Hs with complete R for Fourier transform, R has been changed to nearest cell
    auto Hartree_I_JR_local = comm_map2_first(mpi_comm_global_h.comm, this->HHartree_libri,
                                    Iset_Jset.first, Iset_Jset.second);
    // After comm, each process has complete R for each <I,J> pair, and should be real
    for (const auto &I_Hartree_JR : Hartree_I_JR_local)
    {
        const auto I = I_Hartree_JR.first;
        for (const auto &JR_Hartree : I_Hartree_JR.second)
        {
            const auto J = JR_Hartree.first.first;
            const auto R = JR_Hartree.first.second;
            warn_IJR_nonzero_imag(JR_Hartree.second, I, J, R);
        }
    }
    Profiler::stop("build_real_space_Hartree_5");

    utils::lib_printf("Task %4d: tensor communicate elapsed time: %f\n",
                        mpi_comm_global_h.myid,
                        Profiler::get_wall_time_last("build_real_space_Hartree_5"));
    
    for (int ik = 0; ik < kfrac_target.size(); ik++)
    {
        HHartree_nao_nao.zero_out();
        Profiler::start("build_real_space_Hartree_6", "HHartree IJ -> 2D block");
        const auto &kfrac = kfrac_target[ik];
        const std::function<complex<double>(const int &,
                                            const std::pair<int, std::array<int, 3>> &)>
            fourier =
                [kfrac](const int &I, const std::pair<int, std::array<int, 3>> &J_Ra)
        {
            const auto &Ra = J_Ra.second;
            Vector3<double> R_IJ(Ra[0], Ra[1], Ra[2]);
            const auto ang = (kfrac * R_IJ) * TWO_PI;
            return complex<double>{std::cos(ang), std::sin(ang)};
        };
        collect_block_from_IJ_storage_tensor_transform(
            HHartree_nao_nao, desc_nao_nao, atomic_basis_wfc, atomic_basis_wfc, fourier,
            Hartree_I_JR_local);
        Profiler::stop("build_real_space_Hartree_6");
        
        for (int isp = 0; isp < n_spins; isp++)
        {
            if (this->Hartree_is_ik_nao.count(isp) == 0 ||
                this->Hartree_is_ik_nao[isp].count(ik) == 0)
            {
                this->Hartree_is_ik_nao[isp][ik] =
                    init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
            }
            this->Hartree_is_ik_nao[isp][ik] += HHartree_nao_nao.copy();

            for (int isoc = 0; isoc < n_soc; isoc++)
            {
                Profiler::start("build_real_space_Hartree_7", "Rotate HHartree ij -> KS");        
                const auto &wfc_isp1_k = wfc_target[isp][isoc][ik];
                const auto &wfc_isp2_k = wfc_target[isp][isoc][ik];
                blacs_ctxt_global_h.barrier();
                const auto wfc1_block =
                    get_local_mat(wfc_isp1_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                const auto wfc2_block =
                    get_local_mat(wfc_isp2_k.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                // utils::lib_printf("%s\n", str(wfc_block).c_str());
                // utils::lib_printf("%s\n", desc_nao_nao.info_desc().c_str());
                // utils::lib_printf("%s\n", desc_nband_nao.info_desc().c_str());
                
                ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                            wfc1_block.ptr(), 1, 1, desc_nband_nao.desc,
                                            HHartree_nao_nao.ptr(), 1, 1, desc_nao_nao.desc, 0.0,
                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                ScalapackConnector::pgemm_f(
                    'N', 'C', n_bands, n_bands, n_aos, 1.0, temp_nband_nao.ptr(), 1, 1,
                    desc_nband_nao.desc, wfc2_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                    HHartree_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                Profiler::stop("build_real_space_Hartree_7");

                // collect to master
                Profiler::start("build_real_space_Hartree_8", "Collect EHartree to root process");
                ScalapackConnector::pgemr2d_f(n_bands, n_bands, HHartree_nband_nband.ptr(), 1, 1,
                                                desc_nband_nband.desc, HHartree_nband_nband_fb.ptr(),
                                                1, 1, desc_nband_nband_fb.desc,
                                                desc_nband_nband_fb.ictxt());
                if (this->Hartree_is_ik_KS.count(isp) == 0 ||
                    this->Hartree_is_ik_KS[isp].count(ik) == 0)
                {
                    this->Hartree_is_ik_KS[isp][ik] =
                        init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                }
                this->Hartree_is_ik_KS[isp][ik] += HHartree_nband_nband_fb.copy();
                // for (int ib = 0; ib != n_bands; ib++){
                //     for (int jb = 0; jb != n_bands; jb++){
                //         printf("Hartree_is_ik_KS[%d][%d] (%d,%d) = %e\n", isp, ik, ib, jb,
                //                this->Hartree_is_ik_KS[isp][ik](ib, jb).real());
                //         printf("Hartree_is_ik_KS[%d][%d] (%d,%d) = %e\n", isp, ik, ib, jb,
                //                this->Hartree_is_ik_KS[isp][ik](ib, jb).imag());
                //     }
                // }        
                // print_matrix_mm_file(HHartree_nband_nband_fb,Params::output_dir + "/" + fn, 1e-15);
                // cout << "HHartree_nband_nband_fb isp " << isp  << " ik " << ik << endl <<
                // HHartree_nband_nband_fb;
                if (blacs_ctxt_global_h.myid == 0)
                {
                    for (int ib = 0; ib != n_bands; ib++)
                        this->EHartree[isp][ik][ib] += HHartree_nband_nband_fb(ib, ib).real();
                }
                Profiler::stop("build_real_space_Hartree_8");
            }
        }
    }
}
void Hartree::build_KS_kgrid0()
{
    this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
}
void Hartree::build_KS_kgrid() { this->build_KS(this->mf_.get_eigenvectors(), this->kfrac_list_); }

void Hartree::build_KS_band(const std::vector<std::vector<std::vector<ComplexMatrix>>> &wfc_band,
                        const std::vector<Vector3_Order<double>> &kfrac_band)
{
    this->build_KS(wfc_band, kfrac_band);
}

void Hartree::reset_rspace()
{
    this->HHartree_libri.clear();
    this->is_rspace_build_ = false;
}

void Hartree::reset_kspace()
{
    this->Hartree_is_ik_KS.clear();
    this->EHartree.clear();
    this->is_kspace_built_ = false;
}

}  // namespace LIBRPA
