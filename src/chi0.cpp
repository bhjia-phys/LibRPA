/*! @file: chi0.cpp
 */
#include <iostream>
#include <cstring>
#include "parallel_mpi.h"
#include "profiler.h"
#include "chi0.h"
#include "pbc.h"
#include "ri.h"
#include <omp.h>
#include "matrix.h"
#include "complexmatrix.h"
#include "lapack_connector.h"
#include "input.h"
#include "constants.h"
#include "scalapack_connector.h"
#ifdef __USE_LIBRI
#include <RI/physics/RPA.h>
#endif
#include <array>
#include <map>
//#include "../AtomicTensor/libRI-master/include/RI/physics/RPA.h"
void Chi0::build(const atpair_R_mat_t &LRI_Cs,
                   const vector<Vector3_Order<int>> &Rlist,
                   const Vector3_Order<int> &R_period,
                   const vector<atpair_t> &atpairs_ABF,
                   const vector<Vector3_Order<double>> &qlist,
                   TFGrids::GRID_TYPES gt, bool use_space_time)
{   
    gf_save = gf_discard = 0;
    // reset chi0_q in case the method was called before
    chi0_q.clear();

    // prepare the time-freq grids
    switch (gt)
    {
        case ( TFGrids::GRID_TYPES::Minimax):
        {
            double emin, emax;
            mf.get_E_min_max(emin, emax);
            tfg.generate_minimax(emin, emax);
            break;
        }
        case ( TFGrids::GRID_TYPES::EvenSpaced_TF ):
        {
            // WARN: only used to debug, adjust emin and interval manually
            tfg.generate_evenspaced_tf(0.0304601, 0.0, 0.0116073, 0.0);
            break;
        }
        default:
            throw invalid_argument("chi0 on requested grid is not implemented");
    }

    if (use_space_time && !tfg.has_time_grids())
        throw invalid_argument("current grid type does not have time grids");

    tfg.show();

    if (use_space_time)
    {
        // use space-time method
        for ( auto R: Rlist )
            Rlist_gf.push_back(R);
        for ( auto tau: tfg.get_time_nodes() )
        {
            for ( auto R: Rlist_gf)
            {
                build_gf_Rt(R, tau);
                build_gf_Rt(R, -tau);
            }
        }
        cout << " FINISH GREEN FUN" << endl;
        cout << " Green threshold: " << gf_R_threshold << endl;
        cout << " Green save num: " << gf_save << endl;
        cout << " Green discard num: " << gf_discard << endl;
        build_chi0_q_space_time(LRI_Cs, R_period, atpairs_ABF, qlist);
    }
    else
    {
        // conventional method does not need to build Green's function explicitly
        build_chi0_q_conventional(LRI_Cs, R_period, atpairs_ABF, qlist);
    }
}

void Chi0::build_gf_Rt(Vector3_Order<int> R, double tau)
{
    prof.start("cal_Green_func");
    auto nkpts = mf.get_n_kpoints();
    auto nspins = mf.get_n_spins();
    auto nbands = mf.get_n_bands();
    auto naos = mf.get_n_aos();
    assert (tau != 0);

    // temporary Green's function
    matrix gf_Rt_is_global(naos, naos);

    for (int is = 0; is != nspins; is++)
    {
        gf_Rt_is_global.zero_out();
        auto wg = mf.get_weight()[is];
        if ( tau > 0)
            for (int i = 0; i != wg.size; i++)
            {
                wg.c[i] = 1.0 / nkpts * nspins - wg.c[i];
                if (wg.c[i] < 0) wg.c[i] = 0;
            }
        matrix scale(nkpts, nbands);
        // tau-energy phase
        scale = - 0.5 * tau * (mf.get_eigenvals()[is] - mf.get_efermi());
        /* print_matrix("-(e-ef)*tau", scale); */
        for (int ie = 0; ie != scale.size; ie++)
        {
            // NOTE: enforce non-positive phase
            if ( scale.c[ie] > 0) scale.c[ie] = 0;
            scale.c[ie] = std::exp(scale.c[ie]) * wg.c[ie];
        }
        /* print_matrix("exp(-dE*tau)", scale); */
        for (int ik = 0; ik != nkpts; ik++)
        {
            double ang = - klist[ik] * (R * latvec) * TWO_PI;
            complex<double> kphase = complex<double>(cos(ang), sin(ang));
            /* printf("kphase %f %fj\n", kphase.real(), kphase.imag()); */
            auto scaled_wfc_conj = conj(mf.get_eigenvectors()[is][ik]);
            for ( int ib = 0; ib != nbands; ib++)
                LapackConnector::scal(naos, scale(ik, ib), scaled_wfc_conj.c + naos * ib, 1);
            gf_Rt_is_global += (kphase * transpose(mf.get_eigenvectors()[is][ik], false) * scaled_wfc_conj).real();
        }
        if ( tau < 0 ) gf_Rt_is_global *= -1.;
        for ( int I = 0; I != natom; I++)
        {
            const auto I_num = atom_nw[I];
            for (int J = 0; J != natom; J++)
            {
                const size_t J_num = atom_nw[J];
                matrix tmp_green(I_num, J_num);
                for (size_t i = 0; i != I_num; i++)
                {
                    size_t i_glo = atom_iw_loc2glo(I, i);
                    for (size_t j = 0; j != J_num; j++)
                    {
                        size_t j_glo = atom_iw_loc2glo(J, j);
                        tmp_green(i, j) = gf_Rt_is_global(i_glo, j_glo);
                    }
                }
                if (tmp_green.absmax() > gf_R_threshold)
                {
                    // cout<<" max_green_ele:  "<<tmp_green.absmax()<<endl;
                    gf_is_R_tau[is][I][J][R][tau] = std::move(tmp_green);
                    /* cout << is << " " << I << " " << J << " " << R << " " << tau << endl; */
                    /* print_matrix("gf_is_itau_R[is][I][J][R][itau]", gf_is_itau_R[is][I][J][R][tau]); */
                    gf_save++;
                }
                else
                {
                    gf_discard++;
                }
            }
        }
    }
    prof.stop("cal_Green_func");
}

void Chi0::build_chi0_q_space_time(const atpair_R_mat_t &LRI_Cs,
                                   const Vector3_Order<int> &R_period,
                                   const vector<atpair_t> &atpairs_ABF,
                                   const vector<Vector3_Order<double>> &qlist)
{
    int R_tau_size = Rlist_gf.size() * tfg.size();
    if(use_libri_chi0)
    {
        cout<<"LibRI_routing"<<endl;
#ifdef __USE_LIBRI
        build_chi0_q_space_time_LibRI_routing(LRI_Cs, R_period, atpairs_ABF, qlist);
#else
        cout << "LibRI routing requested, but the executable is not compiled with LibRI" << endl;
        cout << "Please recompiler libRPA with -DUSE_LIBRI and configure include path" << endl;
        throw std::logic_error("compilation");
#endif

    } 
    else if ( atpairs_ABF.size() < R_tau_size )
    {
        cout << "R_tau_routing" << endl;
        para_mpi.chi_parallel_type = Parallel_MPI::parallel_type::R_TAU;
        build_chi0_q_space_time_R_tau_routing(LRI_Cs, R_period, atpairs_ABF, qlist);
    }
    else
    {
        cout << "atom_pair_routing" << endl;
        para_mpi.chi_parallel_type = Parallel_MPI::parallel_type::ATOM_PAIR;
        build_chi0_q_space_time_atom_pair_routing(LRI_Cs, R_period, atpairs_ABF, qlist);
    }
   

}

#ifdef __USE_LIBRI
void Chi0::build_chi0_q_space_time_LibRI_routing(const atpair_R_mat_t &LRI_Cs,
                                                   const Vector3_Order<int> &R_period,
                                                   const vector<atpair_t> &atpairs_ABF,
                                                   const vector<Vector3_Order<double>> &qlist)
{
    prof.start("LibRI_routing");
    map<int,std::array<double,3>> atoms_pos;
    for(int i=0;i!=atom_mu.size();i++)
        atoms_pos.insert(pair<int,std::array<double,3>>{i,{0,0,0}});
    
    std::array<double,3> xa{latvec.e11,latvec.e12,latvec.e13};
    std::array<double,3> ya{latvec.e21,latvec.e22,latvec.e23};
    std::array<double,3> za{latvec.e31,latvec.e32,latvec.e33};
    std::array<std::array<double,3>,3> lat_array{xa,ya,za};

    std::array<int,3> period_array{R_period.x,R_period.y,R_period.z};
    
    RPA<int,int,3,double> rpa;
    rpa.set_parallel(MPI_COMM_WORLD, atoms_pos,lat_array,period_array);
    std::map<int, std::map<std::pair<int,std::array<int,3>>,Tensor<double>>> Cs_libri;
    for(auto &Ip:LRI_Cs)
    {
        auto I=Ip.first;
        map<std::pair<int,std::array<int,3>>,Tensor<double>> Jp_libri;
        for(auto &Jp:Ip.second)
        {
            const auto J=Jp.first;
            for(auto &Rp:Jp.second)
            {
                const auto R=Rp.first;
                std::array<int,3> Ra{R.x,R.y,R.z};
                const auto &mat=Rp.second;
                std::valarray<double> mat_array((*mat).c, (*mat).size);
                std::shared_ptr<std::valarray<double>> mat_ptr = std::make_shared<std::valarray<double>>();
                *mat_ptr=mat_array;
                
                Tensor<double> Tmat({size_t((*mat).nr),size_t((*mat).nc)},mat_ptr);
                Jp_libri.insert(make_pair(make_pair(J,Ra),Tmat));
            }
        }
        Cs_libri.insert(make_pair(I,Jp_libri));
    }
    rpa.set_Cs(Cs_libri,0);

    for (auto it = 0; it != tfg.size(); it++)
    {
        double tau = tfg.get_time_nodes()[it];
        for(auto &isp:gf_is_R_tau)
        {
            std::map<int, std::map<std::pair<int,std::array<int,3>>,Tensor<double>>> gf_po_libri;
            std::map<int, std::map<std::pair<int,std::array<int,3>>,Tensor<double>>> gf_ne_libri;
            for(auto &Ip:isp.second)
            {
                auto I=Ip.first;
                map<std::pair<int,std::array<int,3>>,Tensor<double>> Jp_po_libri;
                map<std::pair<int,std::array<int,3>>,Tensor<double>> Jp_ne_libri;
                for(auto &Jp:Ip.second)
                {
                    auto J=Jp.first;
                    for(auto &Rp:Jp.second)
                    {
                        
                        auto R=Rp.first;
                        std::array<int,3> Ra{R.x,R.y,R.z};
                        auto &taup=Rp.second;

                       // cout<<"I J R:  "<<I<<"  "<<J<<"  "<<R<<endl;
                        const auto &mat_po=taup.at(tau);
                        const auto &mat_ne=taup.at(-tau);

                        std::valarray<double> mat_po_array(mat_po.c, mat_po.size);
                        std::shared_ptr<std::valarray<double>> mat_po_ptr = std::make_shared<std::valarray<double>>();
                        *mat_po_ptr=mat_po_array;
                        Tensor<double> Tmat_po({size_t(mat_po.nr),size_t(mat_po.nc)},mat_po_ptr);

                        std::valarray<double> mat_ne_array(mat_ne.c, mat_ne.size);
                        std::shared_ptr<std::valarray<double>> mat_ne_ptr = std::make_shared<std::valarray<double>>();
                        *mat_ne_ptr=mat_ne_array;
                        Tensor<double> Tmat_ne({size_t(mat_ne.nr),size_t(mat_ne.nc)},mat_ne_ptr);

                        Jp_po_libri.insert(make_pair(make_pair(J,Ra),Tmat_po));
                        Jp_ne_libri.insert(make_pair(make_pair(J,Ra),Tmat_ne));

                    }
                }
                gf_po_libri.insert(make_pair(I,Jp_po_libri));
                gf_ne_libri.insert(make_pair(I,Jp_ne_libri));
            }
            rpa.cal_chi0s(gf_po_libri,gf_ne_libri,0);
        }

    }
    prof.stop("LibRI_routing");
}
#endif


void Chi0::build_chi0_q_space_time_R_tau_routing(const atpair_R_mat_t &LRI_Cs,
                                                 const Vector3_Order<int> &R_period,
                                                 const vector<atpair_t> &atpairs_ABF,
                                                 const vector<Vector3_Order<double>> &qlist)
{
    prof.start("R_tau_routing");
    // taus and Rs to compute on MPI task
    // tend to calculate more Rs on one process
    vector<pair<int, int>> itauiRs_local = dispatcher(0, tfg.size(), 0, Rlist_gf.size(),
                                                      para_mpi.get_myid(), para_mpi.get_size(), true, false);

    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> chi0_q_tmp;
    for ( auto freq: tfg.get_freq_nodes() )
        for ( auto q: qlist)
        {
            for ( auto atpair: atpairs_ABF)
            {
                auto Mu = atpair.first;
                auto Nu = atpair.second;
                chi0_q_tmp[freq][q][Mu][Nu].create(atom_mu[Mu], atom_mu[Nu]);
            }
        }

    omp_lock_t chi0_lock;
    omp_init_lock(&chi0_lock);
    double t_chi0_begin = omp_get_wtime();
    double t_chi0_tot = 0;
#pragma omp parallel for schedule(dynamic)
    for ( auto itau_iR: itauiRs_local)
    {
        auto itau = itau_iR.first;
        auto tau = tfg.get_time_nodes()[itau];
        auto iR = itau_iR.second;
        auto R = Rlist_gf[iR];
        double t_Rtau_begin = omp_get_wtime();
        for ( auto atpair: atpairs_ABF)
        {
            atom_t Mu = atpair.first;
            atom_t Nu = atpair.second;
            for ( int is = 0; is != mf.get_n_spins(); is++ )
            {
                double chi0_ele_begin = omp_get_wtime();
                matrix chi0_tau;
                /* if (itau == 0) // debug first itau */
                    chi0_tau = compute_chi0_s_munu_tau_R(LRI_Cs, R_period, is, Mu, Nu, tau, R);
                /* else continue; // debug first itau */
                double chi0_ele_t = omp_get_wtime() - chi0_ele_begin;
                omp_set_lock(&chi0_lock);
                for ( auto q: qlist )
                {
                    double arg = q * (R * latvec) * TWO_PI;
                    const complex<double> kphase = complex<double>(cos(arg), sin(arg));
                    for ( int ifreq = 0; ifreq != tfg.size(); ifreq++ )
                    {
                        double freq = tfg.get_freq_nodes()[ifreq];
                        double trans = tfg.get_costrans_t2f()(ifreq, itau);
                        const complex<double> weight = trans * kphase;
                        /* cout << weight << endl; */
                        /* cout << complex<double>(cos(arg), sin(arg)) << " * " << trans << " = " << weight << endl; */
                        chi0_q_tmp[freq][q][Mu][Nu] += ComplexMatrix(chi0_tau) * weight;
                    }
                }
                omp_unset_lock(&chi0_lock);
            }
        }
        double t_Rtau_end = omp_get_wtime();
        double time_used = t_Rtau_end - t_Rtau_begin;
        t_chi0_tot += time_used;
        printf("CHI0 p_id: %d,   thread: %d,     R:  ( %d,  %d,  %d ),    tau: %f ,   TIME_USED: %f \n ",
                para_mpi.get_myid(), omp_get_thread_num(), R.x, R.y, R.z, tau, time_used);
    }

    omp_destroy_lock(&chi0_lock);
    // Reduce to MPI
#pragma omp barrier
    for ( int ifreq = 0; ifreq < tfg.size(); ifreq++ )
    {
        double freq = tfg.get_freq_nodes()[ifreq];
        for ( auto q: qlist )
            for ( auto atpair: atpairs_ABF)
            {
                auto Mu = atpair.first;
                auto Nu = atpair.second;
                chi0_q[freq][q][Mu][Nu].create(atom_mu[Mu], atom_mu[Nu]);
                /* cout << "nr/nc chi0_q_tmp: " << chi0_q_tmp[ifreq][iq][Mu][Nu].nr << ", "<< chi0_q_tmp[ifreq][iq][Mu][Nu].nc << endl; */
                /* cout << "nr/nc chi0_q: " << chi0_q[ifreq][iq][Mu][Nu].nr << ", "<< chi0_q[ifreq][iq][Mu][Nu].nc << endl; */
                para_mpi.reduce_ComplexMatrix(chi0_q_tmp[freq][q][Mu][Nu], chi0_q[freq][q][Mu][Nu]);
                /* if (para_mpi.get_myid()==0 && Mu == 0 && Nu == 0 && ifreq == 0 && q == Vector3_Order<double>{0, 0, 0}) */
                /* if (para_mpi.get_myid()==0 && Mu == 0 && Nu == 0 && ifreq == 0 ) */
                /* if (para_mpi.get_myid()==0 && ifreq == 0 && q == Vector3_Order<double>{0, 0, 0}) */
                /* { */
                /*     cout <<  "freq: " << freq << ", q: " << q << endl; */
                /*     printf("Mu %zu Nu %zu\n", Mu, Nu); */
                /*     print_complex_matrix("chi0_q ap, first freq first q", chi0_q[freq][q][Mu][Nu]); */
                /* } */
                chi0_q_tmp[freq][q][Mu].erase(Nu);
            }
    }
    prof.stop("R_tau_routing");
}

void Chi0::build_chi0_q_space_time_atom_pair_routing(const atpair_R_mat_t &LRI_Cs,
                                                 const Vector3_Order<int> &R_period,
                                                 const vector<atpair_t> &atpairs_ABF,
                                                 const vector<Vector3_Order<double>> &qlist)
{
    prof.start("atom_pair_routing");
    auto tot_pair = dispatch_vector(atpairs_ABF, para_mpi.get_myid(), para_mpi.get_size(), false);
    cout << "  tot_pair.size :  " << tot_pair.size() << endl;
    omp_lock_t chi0_lock;
    omp_init_lock(&chi0_lock);
    double t_chi0_begin = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (size_t atom_pair = 0; atom_pair != tot_pair.size(); atom_pair++)
        {
            auto Mu = tot_pair[atom_pair].first;
            auto Nu = tot_pair[atom_pair].second;
            const size_t mu_num = atom_mu[Mu];
            const size_t nu_num = atom_mu[Nu];
            double task_begin = omp_get_wtime();
            map<double, map<Vector3_Order<double>, ComplexMatrix>> tmp_chi0_freq_k;
            for ( auto freq: tfg.get_freq_nodes() )
                for ( auto q: qlist)
                {
                    tmp_chi0_freq_k[freq][q].create(atom_mu[Mu], atom_mu[Nu]);
                }

            for ( int is = 0; is != mf.get_n_spins(); is++ )
                for (auto &R : Rlist_gf)
                {
                    for (auto it = 0; it != tfg.size(); it++)
                    {
                        double tau = tfg.get_time_nodes()[it];
                        ComplexMatrix tmp_chi0_tau(ComplexMatrix(compute_chi0_s_munu_tau_R(LRI_Cs, R_period, is, Mu, Nu, tau, R)));
                        for (auto &q : qlist)
                        {
                            const double arg = (q * (R * latvec)) * TWO_PI;
                            const complex<double> kphase = complex<double>(cos(arg), sin(arg));
                            for (auto ifreq = 0; ifreq != tfg.size(); ifreq++)
                            {
                                double freq = tfg.get_freq_nodes()[ifreq];
                                double trans = tfg.get_costrans_t2f()(ifreq, it);
                                const complex<double> weight = trans * kphase;
                                /* const complex<double> cos_weight_kpashe = kphase * tfg.get_costrans_t2f()[ifreq, it]; */
                                //  cout<<"  tmp_chi0  nr nc:  "<<tmp_chi0_freq_k[ifreq][ik_vec].nr<<"  "<<tmp_chi0_freq_k[ifreq][ik_vec].nc<<"    chi0_tau: "<<tmp_chi0_tau.nr<<"  "<<tmp_chi0_tau.nr<<endl;
                                tmp_chi0_freq_k[freq][q] += tmp_chi0_tau * weight;
                            }
                        }
                    }
                    // vector<matrix> chi0_freq_tmp(tmp_cosine_tran(freq_grid.size(),chi0_tau_tmp));
                }

            double task_end = omp_get_wtime();
            double time_used = task_end - task_begin;
            /* time_task_tot += time_used; */
            omp_set_lock(&chi0_lock);
            for (auto &freq : tfg.get_freq_nodes() )
            {
                for (auto &q : qlist)
                    chi0_q[freq][q][Mu][Nu] = std::move(tmp_chi0_freq_k[freq][q]);
            }
            double add_end = omp_get_wtime();
            double add_time = add_end - task_end;
            printf("CHI0 p_id: %d,   thread: %d,     I: %zu,   J: %zu,   add time: %f  TIME_USED: %f \n ", para_mpi.get_myid(), omp_get_thread_num(), Mu, Nu, add_time, time_used);
            omp_unset_lock(&chi0_lock);
        }
    }
}

matrix Chi0::compute_chi0_s_munu_tau_R(const atpair_R_mat_t &LRI_Cs,
                                       const Vector3_Order<int> &R_period, 
                                       int spin_channel,
                                       atom_t Mu, atom_t Nu, double tau, Vector3_Order<int> R)
{
    prof.start("cal_chi0_element");
    /* printf("     begin chi0  thread: %d,  I: %zu, J: %zu\n",omp_get_thread_num(), Mu, Nu); */

    assert ( tau > 0 );
    // Local RI requires
    const atom_t I_index = Mu;
    const atom_t J_index = Nu;

    const size_t i_num = atom_nw[Mu];
    const size_t mu_num = atom_mu[Mu];

    const size_t j_num = atom_nw[Nu];
    const size_t nu_num = atom_mu[Nu];

    int flag_G_IJRt = 0;
    int flag_G_IJRNt = 0;

    /* printf("     check if already calculated\n"); */
    /* printf("     size of Green_atom: %zu\n", Green_atom.size()); */
    const auto & gf_R_tau = gf_is_R_tau.at(spin_channel);
    if (gf_R_tau.at(I_index).count(J_index))
        if (gf_R_tau.at(I_index).at(J_index).count(R))
        {
            if (gf_R_tau.at(I_index).at(J_index).at(R).count(tau))
                flag_G_IJRt = 1;

            if (gf_R_tau.at(I_index).at(J_index).at(R).count(-tau))
                flag_G_IJRNt = 1;
        }

    matrix X_R2(i_num, j_num * nu_num);
    matrix X_conj_R2(i_num, j_num * nu_num);

    for (const auto &L_pair : Cs[J_index])
    {
        const auto L_index = L_pair.first;
        const size_t l_num = atom_nw[L_index];
        /* printf("     begin is loop\n"); */
        if (gf_R_tau.at(I_index).count(L_index))
        {
            for (const auto &R2_index : L_pair.second)
            {
                const auto R2 = R2_index.first;
                const auto &Cs_mat2 = R2_index.second;

                Vector3_Order<int> R_temp_2(Vector3_Order<int>(R2 + R) % R_period);

                if (gf_R_tau.at(I_index).at(L_index).count(R_temp_2))
                {
                    assert(j_num * l_num == (*Cs_mat2).nr);
                    /* printf("          thread: %d, X_R2 IJL:   %zu,%zu,%zu  R:(  %d,%d,%d  )  tau:%f\n",omp_get_thread_num(),I_index,J_index,L_index,R.x,R.y,R.z,time_tau); */
                    matrix Cs2_reshape(reshape_Cs(j_num, l_num, nu_num, Cs_mat2));
                    
                    if (gf_R_tau.at(I_index).at(L_index).at(R_temp_2).count(tau))
                    {
                        // cout<<"C";
                        prof.start("X");
                        X_R2 += gf_R_tau.at(I_index).at(L_index).at(R_temp_2).at(tau) * Cs2_reshape;
                        prof.stop("X");
                    }

                    if (gf_R_tau.at(I_index).at(L_index).at(R_temp_2).count(-tau))
                    {
                        // cout<<"D";
                        prof.start("X");
                        X_conj_R2 += gf_R_tau.at(I_index).at(L_index).at(R_temp_2).at(-tau) * Cs2_reshape;
                        prof.stop("X");
                    }
                    
                }
            }
        }
    }
    matrix X_R2_rs(reshape_mat(i_num, j_num, nu_num, X_R2));
    matrix X_conj_R2_rs(reshape_mat(i_num, j_num, nu_num, X_conj_R2));
    
    matrix O_sum(mu_num, nu_num);
    for (const auto &K_pair : Cs[I_index])
    {
        const auto K_index = K_pair.first;
        const size_t k_num = atom_nw[K_index];

        for (const auto &R1_index : K_pair.second)
        {
            const auto R1 = R1_index.first;
            const auto &Cs_mat1 = R1_index.second;
            // cout<<"R1:  begin  "<<R1<<endl;
                // matrix X_R2(i_num,j_num*nu_num);
                // matrix X_conj_R2(i_num,j_num*nu_num);
                matrix O(i_num, k_num * nu_num);
                matrix Z(k_num, i_num * nu_num);

                if (flag_G_IJRt || flag_G_IJRNt)
                {
                    matrix N_R2(k_num, j_num * nu_num);
                    matrix N_conj_R2(k_num, j_num * nu_num);
                    for (const auto &L_pair : Cs[J_index])
                    {
                        const auto L_index = L_pair.first;
                        const size_t l_num = atom_nw[L_index];
                        if (gf_R_tau.at(K_index).count(L_index))
                        {
                            for (const auto &R2_index : L_pair.second)
                            {
                                const auto R2 = R2_index.first;
                                const auto &Cs_mat2 = R2_index.second;
                                Vector3_Order<int> R_temp_1(Vector3_Order<int>(R + R2 - R1) % R_period);
                                // Vector3_Order<int> R_temp_2(R2+R);
                                if (gf_R_tau.at(K_index).at(L_index).count(R_temp_1))
                                {
                                    
                                    assert(j_num * l_num == (*Cs_mat2).nr);
                                    // printf("          thread: %d, IJKL:   %d,%d,%d,%d  R:(  %d,%d,%d  )  tau:%f\n",omp_get_thread_num(),I_index,J_index,K_index,L_index,R.x,R.y,R.z,time_tau);
                                    matrix Cs2_reshape(reshape_Cs(j_num, l_num, nu_num, Cs_mat2));
                                    if (flag_G_IJRNt && gf_R_tau.at(K_index).at(L_index).at(R_temp_1).count(tau))
                                    {
                                        // cout<<"A";
                                        prof.start("N");
                                        N_R2 += gf_R_tau.at(K_index).at(L_index).at(R_temp_1).at(tau) * Cs2_reshape;
                                        prof.stop("N");
                                    }
                                    if (flag_G_IJRt && gf_R_tau.at(K_index).at(L_index).at(R_temp_1).count(-tau))
                                    {
                                        // cout<<"B";
                                        prof.start("N");
                                        N_conj_R2 += gf_R_tau.at(K_index).at(L_index).at(R_temp_1).at(-tau) * Cs2_reshape;
                                        prof.stop("N");
                                    }
                                    
                                }
                            }
                        }
                    }
                    if (flag_G_IJRt)
                    {
                        prof.start("O");
                        matrix N_conj_R2_rs(reshape_mat(k_num, j_num, nu_num, N_conj_R2));
                        O += gf_R_tau.at(I_index).at(J_index).at(R).at(tau) * N_conj_R2_rs;
                        prof.stop("O");
                    }
                    if (flag_G_IJRNt)
                    {
                        prof.start("O");
                        matrix N_R2_rs(reshape_mat(k_num, j_num, nu_num, N_R2));
                        O += gf_R_tau.at(I_index).at(J_index).at(R).at(-tau) * N_R2_rs;
                        prof.stop("O");
                    }
                }
                Vector3_Order<int> R_temp_3(Vector3_Order<int>(R - R1) % R_period);
                if (gf_R_tau.at(K_index).count(J_index))
                {
                    if (gf_R_tau.at(K_index).at(J_index).count(R_temp_3))
                    {
                        if (gf_R_tau.at(K_index).at(J_index).at(R_temp_3).count(-tau))
                        {
                            prof.start("Z");
                            Z += gf_R_tau.at(K_index).at(J_index).at(R_temp_3).at(-tau) * X_R2_rs;
                            prof.stop("Z");
                        }

                        if (gf_R_tau.at(K_index).at(J_index).at(R_temp_3).count(tau))
                        {
                            prof.start("Z");
                            Z += gf_R_tau.at(K_index).at(J_index).at(R_temp_3).at(tau) * X_conj_R2_rs;
                            prof.stop("Z");
                        }
                    }
                }

                matrix Z_rs(reshape_mat(k_num, i_num, nu_num, Z));

                O += Z_rs;
                matrix OZ(reshape_mat_21(i_num, k_num, nu_num, O));
                matrix Cs1_tran(transpose(*Cs_mat1));
                prof.start("O");
                O_sum += Cs1_tran * OZ;
                prof.stop("O");
                // cout<<"   K, R1:   "<<K_index<<"   "<<R1;
                // rt_m_max(O_sum);
        }
    }
    
    /* if ( para_mpi.get_myid()==0 && Mu == 1 && Nu == 1 && tau == tfg.get_time_nodes()[0]) */
    /* { */
    /*     cout << R << " Mu=" << Mu << " Nu=" << Nu << " tau:" << tau << endl; */
    /*     print_matrix("space-time chi0", O_sum); */
    /* } */
    return O_sum;
}

void Chi0::build_chi0_q_conventional(const atpair_R_mat_t &LRI_Cs,
                                     const Vector3_Order<int> &R_period,
                                     const vector<atpair_t> &atpair_ABF,
                                     const vector<Vector3_Order<double>> &qlist)
{
    // TODO: low implementation priority
    throw logic_error("Not implemented");
}

/*!
 * @warning ZMY: Since Green's function are currently stored in a real matrix,
 *          the complex conjugate in the equations are in fact not taken.
 *          But the result seems okay for the test cases, compared to FHI-aims.
 *          Definitely should be resolved in the future.
 */
map<size_t, matrix> compute_chi0_munu_tau_LRI_saveN_noreshape(const map<size_t, atom_mapping<matrix>::pair_t_old> &gf_occ_ab_t,
                                                              const map<size_t, atom_mapping<matrix>::pair_t_old> &gf_unocc_ab_t,
                                                              const atpair_R_mat_t &LRI_Cs,
                                                              const vector<Vector3_Order<int>> &Rlist, const Vector3_Order<int> &R_period,
                                                              const vector<int> iRs,
                                                              atom_t Mu, atom_t Nu)
{
    map<size_t, matrix> chi0_tau;

    // Store N and N* so one does not need to recompute for each R of chi
    // the extra memory consumption scales as Cs/nR/natoms,
    // which is a small amount compared to Cs for either small (small natoms, large nR) or large (small nR, large natoms) system
    // N(tau) at R and atom K, at(iR, iK)
    map<pair<Vector3_Order<int>, atom_t>, matrix> N;
    // N*(-tau) at R and atom K, at(iR, iK)
    map<pair<Vector3_Order<int>, atom_t>, matrix> Ncnt; // c -> conjugate, nt -> negative time

    const atom_t iI = Mu;
    const atom_t iJ = Nu;
    const size_t n_i = atom_nw[iI];
    const size_t n_mu = atom_mu[Mu];
    const size_t n_j = atom_nw[iJ];
    const size_t n_nu = atom_mu[Nu];

    // pre-compute N and N*
    // only need to calculate N on R-R1, with R in Green's function and R1 from Cs
    // TODO: may reuse the N code to compute N*, by parsing G(t) and G(-t) in same map as done by Shi Rong
    // TODO: decouple N and N*, reduce memory usage
    cout << "Mu: " << Mu << ", Nu: " << Nu << endl;
    for ( auto iR: iRs )
    {
        /* cout << "iR: " << iR << endl; */
        for ( auto const & iK_R1_Cs: LRI_Cs.at(Mu) )
        {
            auto iK = iK_R1_Cs.first;
            /* cout << "iK: " << iK << endl; */
            /* cout << "iR: " << iR << ", iK: " << iK << endl; */
            const size_t n_k = atom_nw[iK];
            for ( auto const & R1_Cs: iK_R1_Cs.second )
            {
                auto const & R1 = R1_Cs.first;
                auto const & RmR1 = Vector3_Order<int>(Rlist[iR] - R1);
                /* cout << "iRmR1: " << iRmR1 << endl; */
                /* cout << "Testing N.count(iRmR1, iK): " << N.count({iRmR1, iK}) << endl; */
                /* cout << "Testing gf_occ_ab_t.count(iR): " << gf_occ_ab_t.count(iR) << endl; */
                // N part, do not recompute and should have the GF counterpart
                if ( N.count({RmR1, iK}) == 0 && gf_occ_ab_t.count(iR) )
                {
                    N[{RmR1, iK}].create(n_nu, n_j*n_k);
                    matrix & N_iRiK = N[{RmR1, iK}];
                    for (auto const & iL_R2_Cs: LRI_Cs.at(Nu))
                    {
                        auto const & iL = iL_R2_Cs.first;
                        const size_t n_l = atom_nw[iL];
                        for ( auto const & R2_Cs: iL_R2_Cs.second )
                        {
                            auto const & R2 = R2_Cs.first;
                            auto mRpR1mR2 = Vector3_Order<int>(-RmR1-R2) % R_period;
                            int i_mRpR1mR2 = get_R_index(Rlist, mRpR1mR2);
                            if ( gf_unocc_ab_t.count(i_mRpR1mR2) )
                            {
                                if (gf_unocc_ab_t.at(i_mRpR1mR2).count(iL) == 0 ||
                                    gf_unocc_ab_t.at(i_mRpR1mR2).at(iL).count(iK) == 0) continue;
                                const matrix & gf_unocc = gf_unocc_ab_t.at(i_mRpR1mR2).at(iL).at(iK);
                                auto const & Cs_nu_jlR2 = R2_Cs.second;
                                assert( Cs_nu_jlR2->nr == n_j * n_l && Cs_nu_jlR2->nc == n_nu );
                                matrix tran_Cs_nu_jlR2 = transpose(*Cs_nu_jlR2);
                                assert( gf_unocc.nr == n_l && gf_unocc.nc == n_k );
                                for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                                    LapackConnector::gemm('N', 'N', n_j, n_k, n_l, 1.0,
                                                          tran_Cs_nu_jlR2.c+i_nu*n_j*n_l, n_l,
                                                          gf_unocc.c, gf_unocc.nc,
                                                          1.0, N_iRiK.c + i_nu*n_j*n_k, n_k);
                            }
                        }
                    }
                }
                // N* part
                if ( !Ncnt.count({RmR1, iK}) && gf_unocc_ab_t.count(iR) )
                {
                    Ncnt[{RmR1, iK}].create(n_nu, n_j*n_k);
                    matrix & Ncnt_iRiK = Ncnt.at({RmR1, iK});
                    for (auto const & iL_R2_Cs: LRI_Cs.at(Nu))
                    {
                        auto const & iL = iL_R2_Cs.first;
                        const size_t n_l = atom_nw[iL];
                        for ( auto const & R2_Cs: iL_R2_Cs.second )
                        {
                            auto const & R2 = R2_Cs.first;
                            auto mRpR1mR2 = Vector3_Order<int>(-RmR1-R2) % R_period;
                            int i_mRpR1mR2 = get_R_index(Rlist, mRpR1mR2);
                            if ( gf_occ_ab_t.count(i_mRpR1mR2) )
                            {
                                if (gf_occ_ab_t.at(i_mRpR1mR2).count(iL) == 0 ||
                                    gf_occ_ab_t.at(i_mRpR1mR2).at(iL).count(iK) == 0) continue;
                                const matrix & gf_occ = gf_occ_ab_t.at(i_mRpR1mR2).at(iL).at(iK);
                                auto const & Cs_nu_jlR2 = R2_Cs.second;
                                assert( Cs_nu_jlR2->nr == n_j * n_l && Cs_nu_jlR2->nc == n_nu );
                                matrix tran_Cs_nu_jlR2 = transpose(*Cs_nu_jlR2);
                                assert( gf_occ.nr == n_l && gf_occ.nc == n_k );
                                for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                                    LapackConnector::gemm('N', 'N', n_j, n_k, n_l, 1.0,
                                                          tran_Cs_nu_jlR2.c+i_nu*n_j*n_l, n_l,
                                                          gf_occ.c, n_k,
                                                          1.0, Ncnt_iRiK.c + i_nu*n_j*n_k, n_k);
                            }
                        }
                    }
                }
            }
        }
    }
    cout << "Size N: " << N.size() << endl;
    /* cout << "Precalcualted N, size (iR1 x iK): " << N.size() << endl; */

    for ( auto iR: iRs )
    {
        chi0_tau[iR].create(n_mu, n_nu); // TODO: selectively create by G

        for ( auto const & iK_R1_Cs: LRI_Cs.at(Mu) )
        {
            auto const iK = iK_R1_Cs.first;
            const size_t n_k = atom_nw[iK];
            for (auto const & R1_Cs: iK_R1_Cs.second)
            {
                const auto & Cs = R1_Cs.second;
                // a temporary matrix to store the sum of M(t) and M*(-t)
                matrix MpMc(n_nu, n_i*n_k);
                auto RmR1 = Vector3_Order<int>(Rlist[iR]-R1_Cs.first);
                // M(t)=G(t)N(t)
                if ( gf_occ_ab_t.count(iR) && N.count({RmR1, iK}))
                {
                    if ( gf_occ_ab_t.at(iR).count(iI) != 0 &&
                         gf_occ_ab_t.at(iR).at(iI).count(iJ) != 0)
                    {
                        /* cout << "Computing GN" << endl; */
                        const matrix & gf = gf_occ_ab_t.at(iR).at(iI).at(iJ);
                        const matrix & _N = N.at({RmR1, iK});
                        assert( _N.nr == n_nu && _N.nc == n_j * n_k);
                        for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                            LapackConnector::gemm('N', 'N', n_i, n_k, n_j, 1.0,
                                                  gf.c, n_j,
                                                  _N.c+i_nu*n_j*n_k, n_k,
                                                  1.0, MpMc.c+i_nu*n_i*n_k, n_k);
                    }
                }
                // M*(-t)=G*(-t)N*(-t)
                if ( gf_unocc_ab_t.count(iR) && N.count({RmR1, iK}))
                {
                    if ( gf_unocc_ab_t.at(iR).count(iI) != 0 &&
                         gf_unocc_ab_t.at(iR).at(iI).count(iJ) != 0)
                    {
                        /* cout << "Computing G*N*" << endl; */
                        const matrix & gf = gf_unocc_ab_t.at(iR).at(iI).at(iJ);
                        const matrix & _Ncnt = Ncnt.at({RmR1, iK});
                        assert( _Ncnt.nr == n_nu && _Ncnt.nc == n_j * n_k);
                        for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                            LapackConnector::gemm('N', 'N', n_i, n_k, n_j, 1.0,
                                                  gf.c, n_j,
                                                  _Ncnt.c+i_nu*n_j*n_k, n_k,
                                                  1.0, MpMc.c+i_nu*n_i*n_k, n_k);
                    }
                }
                LapackConnector::gemm('T', 'T', n_mu, n_nu, n_i*n_k, 1.0,
                        Cs->c, n_mu,
                        MpMc.c, n_i*n_k, 1.0, chi0_tau[iR].c, n_nu);
            }
        }
    }

    // clean up N to free memory, as they have finished their job
    N.clear();
    Ncnt.clear();
    
    for ( auto iR: iRs )
    {
        matrix X(n_nu, n_j*n_i), Ynt(n_nu, n_i*n_j);
        matrix Xcnt(n_nu, n_j*n_i), Yc(n_nu, n_i*n_j);
        // X
        for ( auto const & iL_R2_Cs: LRI_Cs.at(Nu) )
        {
            auto const iL = iL_R2_Cs.first;
            auto const n_l = atom_nw[iL];
            for ( auto & R2_Cs: iL_R2_Cs.second )
            {
                auto const R2 = R2_Cs.first;
                auto const & Cs = R2_Cs.second;
                auto mR2mR = Vector3_Order<int>(-Rlist[iR]-R2) % R_period;
                auto i_mR2mR = get_R_index(Rlist, mR2mR);
                if ( gf_unocc_ab_t.count(i_mR2mR) &&
                     gf_unocc_ab_t.at(i_mR2mR).count(iL) &&
                     gf_unocc_ab_t.at(i_mR2mR).at(iL).count(iI) )
                {
                    const auto tran_Cs_nu_jl = transpose(*Cs);
                    auto const & gfu = gf_unocc_ab_t.at(i_mR2mR).at(iL).at(iI);
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        LapackConnector::gemm('N', 'N', n_j, n_i, n_l, 1.0,
                                              tran_Cs_nu_jl.c+i_nu*n_j*n_l, n_l,
                                              gfu.c, n_i,
                                              1.0, X.c+i_nu*n_j*n_i, n_i);
                }
                if ( gf_occ_ab_t.count(i_mR2mR) &&
                     gf_occ_ab_t.at(i_mR2mR).count(iL) &&
                     gf_occ_ab_t.at(i_mR2mR).at(iL).count(iI) )
                {
                    const auto tran_Cs_nu_jl = transpose(*Cs);
                    // gf shall take conjugate here
                    auto const & gf = gf_occ_ab_t.at(i_mR2mR).at(iL).at(iI);
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        LapackConnector::gemm('N', 'N', n_j, n_i, n_l, 1.0,
                                              tran_Cs_nu_jl.c+i_nu*n_j*n_l, n_l,
                                              gf.c, n_i,
                                              1.0, Xcnt.c+i_nu*n_j*n_i, n_i);
                }
            }
        }
        // Y
        for ( auto const & iK_R1_Cs: LRI_Cs.at(Mu) )
        {
            auto const iK= iK_R1_Cs.first;
            auto const n_k = atom_nw[iK];
            for ( auto & R1_Cs: iK_R1_Cs.second )
            {
                auto const R1 = R1_Cs.first;
                auto const Cs = R1_Cs.second;
                auto RmR1 = Vector3_Order<int>(Rlist[iR]-R1) % R_period;
                auto i_RmR1 = get_R_index(Rlist, RmR1);
                if ( gf_occ_ab_t.count(i_RmR1) &&
                     gf_occ_ab_t.at(i_RmR1).count(iK) &&
                     gf_occ_ab_t.at(i_RmR1).at(iK).count(iJ) )
                {
                    const matrix tran_Cs_mu_ik = transpose(*Cs);
                    auto const & gf = gf_occ_ab_t.at(i_RmR1).at(iK).at(iJ);
                    for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                        // transpose so that result stored in ji order
                        LapackConnector::gemm('T', 'T', n_j, n_i, n_k, 1.0,
                                              gf.c, n_j,
                                              tran_Cs_mu_ik.c+i_mu*n_i*n_k, n_k,
                                              1.0, Ynt.c+i_mu*n_j*n_i, n_i);
                }
                if ( gf_unocc_ab_t.count(i_RmR1) &&
                     gf_unocc_ab_t.at(i_RmR1).count(iK) &&
                     gf_unocc_ab_t.at(i_RmR1).at(iK).count(iJ) )
                {
                    const matrix tran_Cs_mu_ik = transpose(*Cs);
                    auto const & gfu = gf_unocc_ab_t.at(i_RmR1).at(iK).at(iJ);
                    for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                        // transpose so that result stored in ji order
                        LapackConnector::gemm('T', 'T', n_j, n_i, n_k, 1.0,
                                              gfu.c, n_j,
                                              tran_Cs_mu_ik.c+i_mu*n_i*n_k, n_k,
                                              1.0, Yc.c+i_mu*n_j*n_i, n_i);
                }
            }
        }
        LapackConnector::gemm('N', 'T', n_mu, n_nu, n_i*n_j, 1.0,
                Ynt.c, n_i*n_j,
                X.c, n_i*n_j, 1.0, chi0_tau[iR].c, n_nu);
        LapackConnector::gemm('N', 'T', n_mu, n_nu, n_i*n_j, 1.0,
                Yc.c, n_i*n_j,
                Xcnt.c, n_i*n_j, 1.0, chi0_tau[iR].c, n_nu);
    }
    /* if (para_mpi.get_myid()==0 && Mu == 0 && Nu == 0) */
    if (para_mpi.get_myid()==0 && Mu == 0 && Nu == 1)
    {
        /* for (auto iR_chi0_tau: chi0_tau) */
        /* { */
            /* if (iR_chi0_tau.first!=4) continue; */
            /* cout << Rlist[iR_chi0_tau.first] << " Mu=" << Mu << " Nu=" << Nu; */
            /* print_matrix("chi tauR at first tau and R", iR_chi0_tau.second); */
        /* } */
    }
    /* cout << "Done real-space chi0" << endl; */

    return chi0_tau;
}

bool use_libri_chi0 = false;
