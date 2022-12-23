#include <algorithm>
#include <array>
#include <valarray>
#include "epsilon.h"
#include <math.h>
#include <omp.h>
#include "input.h"
#include "ri.h"
#include "matrix_m_parallel_utils.h"
#include "params.h"
#include "parallel_mpi.h"
#include "pbc.h"
#include "lapack_connector.h"
#include "ri.h"
#include "scalapack_connector.h"
#include "libri_utils.h"
#ifdef __USE_LIBRI
#include <RI/global/Tensor.h>
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#endif
// #include "print_stl.h"
#include <omp.h>

using LIBRPA::mpi_comm_world_h;
using LIBRPA::Array_Desc;

complex<double> compute_pi_det_blacs(ComplexMatrix &loc_piT, const Array_Desc &arrdesc_pi, int *ipiv, int &info)
{
    // int range_all = atom_mu_part_range[natom-1]+atom_mu[natom-1];

    // int desc_pi[9];
    // int loc_row, loc_col, info;
    // int row_nblk=1;
    // int col_nblk=1;
    int one=1;
    int range_all= N_all_mu;
    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,range_all,range_all,row_nblk,col_nblk);
    // int *ipiv = new int [loc_row*10];
    // ComplexMatrix loc_piT(loc_col,loc_row);
    
    // for(int i=0;i!=loc_row;i++)
    // {
    //     int global_row = para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
    //     int mu;
    //     int I=atom_mu_glo2loc(global_row,mu);
    //     for(int j=0;j!=loc_col;j++)
    //     {
    //         int global_col = para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
    //         int nu;
    //         int J=atom_mu_glo2loc(global_col,nu);
            
    //         if( global_col == global_row)
    //         {
    //             loc_piT(j,i)=complex<double>(1.0,0.0) - pi_freq_q.at(I).at(J)(mu,nu);
    //         }
    //         else
    //         {
    //             loc_piT(j,i)=-1*  pi_freq_q.at(I).at(J)(mu,nu);
    //         }
            
    //     }
    // }
    int DESCPI_T[9];
    // if(out_pi)
    // {
    //     print_complex_real_matrix("first_pi",pi_freq_q.at(0).at(0));
    //     print_complex_real_matrix("first_loc_piT_mat",loc_piT);
    // }
    
    ScalapackConnector::transpose_desc(DESCPI_T, arrdesc_pi.desc);

   // para_mpi.mpi_barrier();
    //printf("   before LU Myid: %d        Available DOS memory = %ld bytes\n",mpi_comm_world_h.myid, memavail());
    //printf("   before LU myid: %d  range_all: %d,  loc_mat.size: %d\n",mpi_comm_world_h.myid,range_all,loc_piT.size);
    pzgetrf_(&range_all,&range_all,loc_piT.c,&one,&one,DESCPI_T,ipiv, &info);
    //printf("   after LU myid: %d\n",mpi_comm_world_h.myid);
    complex<double> ln_det_loc(0.0,0.0);
    complex<double> ln_det_all(0.0,0.0);
    for(int ig=0;ig!=range_all;ig++)
    {
        // int locr=para_mpi.localIndex(ig,row_nblk,para_mpi.nprow,para_mpi.myprow);
		// int locc=para_mpi.localIndex(ig,col_nblk,para_mpi.npcol,para_mpi.mypcol);
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            // if(ipiv[locr]!=(ig+1))
            // 	det_loc=-1*det_loc * loc_piT(locc,locr);
            // else
            // 	det_loc=det_loc * loc_piT(locc,locr);
            if(loc_piT(locc,locr).real()>0)
                ln_det_loc+=std::log(loc_piT(locc,locr));
            else
                ln_det_loc+=std::log(-loc_piT(locc,locr));
		}
    }
    MPI_Allreduce(&ln_det_loc,&ln_det_all,1,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
    return ln_det_all;
}


CorrEnergy compute_RPA_correlation_blacs(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    // if(para_mpi.chi_parallel_type == Parallel_MPI::parallel_type::R_TAU)
    // {
    //     compute_RPA_correlation(chi0,coulmat);
    //     //return corr;
    // }
    if (mpi_comm_world_h.myid == 0)
        printf("Calculating EcRPA with BLACS/ScaLAPACK\n");
    // printf("Calculating EcRPA with BLACS, pid:  %d\n", mpi_comm_world_h.myid);
    const auto & mf = chi0.mf;

    //init blacs pi_mat
    int np_row= int(sqrt(mpi_comm_world_h.nprocs));
    int np_col= int(mpi_comm_world_h.nprocs/np_row);
    int desc_pi[9];
    int row_nblk=ceil(N_all_mu/np_row);
    int col_nblk=ceil(N_all_mu/np_col);
    if(LIBRPA::mpi_comm_world_h.is_root())
        printf("| BLACS row_nblk: %d,  col_nblk: %d\n",row_nblk,col_nblk);
    int one=1;
    LIBRPA::Array_Desc arrdesc_pi(LIBRPA::blacs_ctxt_world_h);
    arrdesc_pi.init(N_all_mu, N_all_mu, row_nblk, col_nblk, 0, 0);
    int loc_row = arrdesc_pi.m_loc(), loc_col = arrdesc_pi.n_loc(), info;

    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,N_all_mu,N_all_mu,row_nblk,col_nblk);
    int *ipiv = new int [loc_row*10];
    

    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    for (const auto &freq_q_MuNuchi0 : chi0.get_chi0_q())
    {
        const auto freq = freq_q_MuNuchi0.first;
        const double freq_weight = chi0.tfg.find_freq_weight(freq);
        for (const auto &q_MuNuchi0 : freq_q_MuNuchi0.second)
        {
            double task_begin = omp_get_wtime();
            const auto q = q_MuNuchi0.first;
            auto &MuNuchi0 = q_MuNuchi0.second;
            
            ComplexMatrix loc_piT(loc_col,loc_row);
            complex<double> trace_pi(0.0,0.0);
            for(int Mu=0;Mu!=natom;Mu++)
            {
               // printf(" |process %d,  Mu:  %d\n",mpi_comm_world_h.myid,Mu);
                const size_t n_mu = atom_mu[Mu];
                atom_mapping<ComplexMatrix>::pair_t_old Vq_row=gather_vq_row_q(Mu,coulmat,q);
                //printf("   |process %d, Mu: %d  vq_row.size: %d\n",para_mpi.get_myid(),Mu,Vq_row[Mu].size());
                ComplexMatrix loc_pi_row=compute_Pi_freq_q_row(q,MuNuchi0,Vq_row,Mu);
                //printf("   |process %d,   compute_pi\n",para_mpi.get_myid());
                ComplexMatrix glo_pi_row( n_mu,N_all_mu);
                mpi_comm_world_h.barrier();
                mpi_comm_world_h.allreduce_ComplexMatrix(loc_pi_row,glo_pi_row);
                //cout<<"  glo_pi_rowT nr,nc: "<<glo_pi_row.nr<<" "<<glo_pi_row.nc<<endl; 
                
                for(int i_mu=0;i_mu!=n_mu;i_mu++)
                    trace_pi+=glo_pi_row(i_mu,atom_mu_part_range[Mu]+i_mu);
                //select glo_pi_rowT to pi_blacs 
                for(int i=0;i!=loc_row;i++)
                {
                    // int global_row = para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
                    int global_row = arrdesc_pi.indx_l2g_r(i);
                    int mu_blacs;
                    int I_blacs=atom_mu_glo2loc(global_row,mu_blacs);
                    if(I_blacs== Mu)
                        for(int j=0;j!=loc_col;j++)
                        {
                            // int global_col = para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
                            int global_col = arrdesc_pi.indx_l2g_c(j);
                            int nu_blacs;
                            int J_blacs=atom_mu_glo2loc(global_col,nu_blacs);
                            //cout<<" Mu: "<<Mu<<"  i,j: "<<i<<"  "<<j<<"    glo_row,col: "<<global_row<<"  "<<global_col<<"  J:"<<J_blacs<< "  index i,j: "<<atom_mu_part_range[J_blacs] + mu_blacs<<" "<<nu_blacs<<endl;
                            if( global_col == global_row)
                            {
                                loc_piT(j,i)=complex<double>(1.0,0.0) - glo_pi_row(mu_blacs, atom_mu_part_range[J_blacs]+nu_blacs);
                            }
                            else
                            {
                                loc_piT(j,i) = -glo_pi_row(mu_blacs, atom_mu_part_range[J_blacs]+nu_blacs);
                            }

                        }

                }
                
            }
            double task_mid = omp_get_wtime();
            //printf("|process  %d, before det\n",mpi_comm_world_h.myid);
            complex<double> ln_det=compute_pi_det_blacs(loc_piT, arrdesc_pi, ipiv, info);
            double task_end = omp_get_wtime();
            if(mpi_comm_world_h.is_root())
                printf("| After det for freq:  %f,  q: ( %f, %f, %f)   TIME_LOCMAT: %f   TIME_DET: %f\n",freq, q.x,q.y,q.z,task_mid-task_begin,task_end-task_mid);
            //para_mpi.mpi_barrier();
            if(mpi_comm_world_h.myid==0)
            {
                complex<double> rpa_for_omega_q=trace_pi+ln_det;
                //cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "      lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;//!check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
            }
        }
    }
    
    if(mpi_comm_world_h.myid==0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            //cout << q_crpa.first << q_crpa.second << endl;
        }
        //cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)    <<tot_RPA_energy << endl;
    }
    mpi_comm_world_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    if (mpi_comm_world_h.myid == 0)
        printf("Calculating EcRPA without BLACS/ScaLAPACK\n");
    // printf("Begin cal cRPA , pid:  %d\n", mpi_comm_world_h.myid);
    const auto & mf = chi0.mf;
    
    // freq, q
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi_freq_q_Mu_Nu;
    if (LIBRPA::chi_parallel_type == LIBRPA::parallel_type::ATOM_PAIR)
        pi_freq_q_Mu_Nu = compute_Pi_q_MPI(chi0, coulmat);
    else 
        pi_freq_q_Mu_Nu = compute_Pi_q(chi0, coulmat);
    printf("Finish Pi freq on Proc %4d, size %zu\n", mpi_comm_world_h.myid, pi_freq_q_Mu_Nu.size());
    mpi_comm_world_h.barrier();

    int range_all = N_all_mu;

    
    vector<int> part_range;
    part_range.resize(atom_mu.size());
    part_range[0] = 0;
    int count_range = 0;
    for (int I = 0; I != atom_mu.size() - 1; I++)
    {
        count_range += atom_mu[I];
        part_range[I + 1] = count_range;
    }

    // cout << "part_range:" << endl;
    // for (int I = 0; I != atom_mu.size(); I++)
    // {
    //     cout << part_range[I] << endl;
    // }
    // cout << "part_range over" << endl;

    // pi_freq_q contains all atoms
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
    
    for (const auto &freq_q_MuNupi : pi_freq_q_Mu_Nu)
    {
        const auto freq = freq_q_MuNupi.first;

        for (const auto &q_MuNupi : freq_q_MuNupi.second)
        {
            const auto q = q_MuNupi.first;
            const auto MuNupi = q_MuNupi.second;
            pi_freq_q[freq][q].create(range_all, range_all);
    
            ComplexMatrix pi_munu_tmp(range_all, range_all);
            pi_munu_tmp.zero_out();

            for (const auto &Mu_Nupi : MuNupi)
            {
                const auto Mu = Mu_Nupi.first;
                const auto Nupi = Mu_Nupi.second;
                const size_t n_mu = atom_mu[Mu];
                for (const auto &Nu_pi : Nupi)
                {
                    const auto Nu = Nu_pi.first;
                    const auto pimat = Nu_pi.second;
                    const size_t n_nu = atom_mu[Nu];
    
                    for (size_t mu = 0; mu != n_mu; ++mu)
                    {
                        for (size_t nu = 0; nu != n_nu; ++nu)
                        {
                            pi_munu_tmp(part_range[Mu] + mu, part_range[Nu] + nu) += pimat(mu, nu);
                        }
                    }
                }
            }
            if (LIBRPA::chi_parallel_type == LIBRPA::parallel_type::ATOM_PAIR)
            {
                mpi_comm_world_h.reduce_ComplexMatrix(pi_munu_tmp, pi_freq_q.at(freq).at(q), 0);
            }
            else
            {
                pi_freq_q.at(freq).at(q) = std::move(pi_munu_tmp);
            }
        }
    }
    // mpi_comm_world_h.barrier();
    if (mpi_comm_world_h.myid == 0)
    {
        complex<double> tot_RPA_energy(0.0, 0.0);
        map<Vector3_Order<double>, complex<double>> cRPA_q;
        for (const auto &freq_qpi : pi_freq_q)
        {
            const auto freq = freq_qpi.first;
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            for (const auto &q_pi : freq_qpi.second)
            {
                const auto q = q_pi.first;
                const auto pimat = q_pi.second;
                complex<double> rpa_for_omega_q(0.0, 0.0);
                ComplexMatrix identity(range_all, range_all);
                ComplexMatrix identity_minus_pi(range_all, range_all);
    
                identity.set_as_identity_matrix();
    
                identity_minus_pi = identity - pi_freq_q[freq][q];
                complex<double> det_for_rpa(1.0, 0.0);
                int info_LU = 0;
                int *ipiv = new int[range_all];
                LapackConnector::zgetrf(range_all, range_all, identity_minus_pi, range_all, ipiv, &info_LU);
                for (int ib = 0; ib != range_all; ib++)
                {
                    if (ipiv[ib] != (ib + 1))
                        det_for_rpa = -det_for_rpa * identity_minus_pi(ib, ib);
                    else
                        det_for_rpa = det_for_rpa * identity_minus_pi(ib, ib);
                }
                delete[] ipiv;
    
                complex<double> trace_pi;
                complex<double> ln_det;
                ln_det = std::log(det_for_rpa);
                trace_pi = trace(pi_freq_q.at(freq).at(q));
                // cout << "PI trace vector:" << endl;
                // cout << endl;
                rpa_for_omega_q = ln_det + trace_pi;
                // cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "      lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * irk_weight[q] / TWO_PI;
                tot_RPA_energy += rpa_for_omega_q * freq_weight * irk_weight[q]  / TWO_PI;
            }
        }
    
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
        corr.value = tot_RPA_energy;
    }
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_MP2_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    corr.etype = CorrEnergy::type::MP2;
    return corr;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    // printf("Begin cal_pi_k , pid:  %d\n", mpi_comm_world_h.myid);
    for (auto const & freq_qJQchi0: chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second )
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = atom_mu[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = atom_mu[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (int I=0;I!=natom;I++)
                    {
                        //const size_t I = I_p.first;
                        const size_t I_mu = atom_mu[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q)
                            pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             printf("  |process  %d, pi atpair: %d, %d \n",mpi_comm_world_h.myid,Ip.first,Jp.first);
        }
        
    }

    // ofstream fp;
    // std::stringstream ss;
    // ss<<"out_pi_rank_"<<mpi_comm_world_h.myid<<".txt";
    // fp.open(ss.str());
    for (auto &freq_p : chi0.get_chi0_q())
    {
        const double freq = freq_p.first;
        const auto chi0_freq = freq_p.second;
        for (auto &k_pair : chi0_freq)
        {
            Vector3_Order<double> ik_vec = k_pair.first;
            auto chi0_freq_k = k_pair.second;
            for (auto &J_p : chi0_freq_k)
            {
                const size_t J = J_p.first;
                for (auto &Q_p : J_p.second)
                {
                    const size_t Q = Q_p.first;
                    auto &chi0_mat = Q_p.second;
                    for (int I=0;I!=natom;I++)
                    {
                        //const size_t I = I_p.first;
                        //printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", mpi_comm_world_h.myid, I, J, Q);
                        //  cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);
                        if (I <= J)
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J, Q,1);
                            //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) += (*Vq.at(I).at(J).at(ik_vec)) * chi0_mat;
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            // {
                            //     std:stringstream sm;
                            //     complex<double> trace_pi;
                            //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                            //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " << ik_vec << "  trace_pi:  " << trace_pi << endl;
                            //     print_complex_matrix_file(sm.str().c_str(), (*Vq.at(I).at(J).at(ik_vec)),fp,false);
                            //     print_complex_matrix_file("chi0:", chi0_mat,fp,false);
                            //     print_complex_matrix_file("pi_mat:", pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                            //}
                        }
                        else
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J, Q,2);
                            //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) += transpose(*Vq.at(J).at(I).at(ik_vec), 1) * chi0_mat;
                        }

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            if (I <= Q)
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J, Q,3);
                                //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) += (*Vq.at(I).at(Q).at(ik_vec)) * chi0_QJ;
                            }
                            else
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J, Q,4);
                                //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0, 0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) += transpose(*Vq.at(Q).at(I).at(ik_vec), 1) * chi0_QJ;
                            }
                        }
                    }
                }
            }
        }
    }
    //fp.close();
    // print_complex_matrix(" first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0)); 
    /* print_complex_matrix("  last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    return pi;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q_MPI(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    // printf("Begin cal_pi_k , pid:  %d\n", mpi_comm_world_h.myid);
    for (auto const & freq_qJQchi0: chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second )
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = atom_mu[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = atom_mu[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (int I=0;I!=natom;I++)
                    {
                        //const size_t I = I_p.first;
                        const size_t I_mu = atom_mu[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q)
                            pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             printf("  |process  %d, pi atpair: %d, %d \n",mpi_comm_world_h.myid,Ip.first,Jp.first);
        }
        
    }

    // ofstream fp;
    // std::stringstream ss;
    // ss<<"out_pi_rank_"<<mpi_comm_world_h.myid<<".txt";
    // fp.open(ss.str());
    for (auto &k_pair : irk_weight)
    {
        Vector3_Order<double> ik_vec = k_pair.first;
        for (int I=0;I!=natom;I++)
        {
            atom_mapping<ComplexMatrix>::pair_t_old Vq_row=gather_vq_row_q(I,coulmat,ik_vec);
            for (auto &freq_p : chi0.get_chi0_q())
            {
                const double freq = freq_p.first;
                const auto chi0_freq = freq_p.second;
            
                
                auto chi0_freq_k = freq_p.second.at(ik_vec);

                for (auto &J_p : chi0_freq_k)
                {
                    const size_t J = J_p.first;
                    for (auto &Q_p : J_p.second)
                    {
                        const size_t Q = Q_p.first;
                        auto &chi0_mat = Q_p.second;
                    
                        //const size_t I = I_p.first;
                        //printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", mpi_comm_world_h.myid, I, J, Q);
                        //  cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);
                        
                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J, Q,1);
                        //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                        pi.at(freq).at(ik_vec).at(I).at(Q) += Vq_row.at(I).at(J) * chi0_mat;
                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        // {
                        //     std:stringstream sm;
                        //     complex<double> trace_pi;
                        //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                        //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " << ik_vec << "  trace_pi:  " << trace_pi << endl;
                        //     print_complex_matrix_file(sm.str().c_str(), Vq_row.at(I).at(J),fp,false);
                        //     print_complex_matrix_file("chi0:", chi0_mat,fp,false);
                        //     print_complex_matrix_file("pi_mat:", pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                        // }  

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", mpi_comm_world_h.myid, I, J,Q,3);
                            //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
                        }
                    }
                }
            }
        }
    }
    //fp.close();
    // print_complex_matrix(" first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0)); 
    /* print_complex_matrix("  last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    return pi;
}



ComplexMatrix compute_Pi_freq_q_row(const Vector3_Order<double> &ik_vec, const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q, const atom_mapping<ComplexMatrix>::pair_t_old &Vq_row, const int &I)
{
    map<size_t,ComplexMatrix> pi;
    // printf("Begin cal_pi_k , pid:  %d\n", mpi_comm_world_h.myid);
    auto I_mu=atom_mu[I];
    for(int J=0;J!=natom;J++)
        pi[J].create(I_mu,atom_mu[J]);

omp_lock_t pi_lock;
omp_init_lock(&pi_lock);
#pragma omp parallel for schedule(dynamic)
    for (int iap=0;iap!=local_atpair.size();iap++)
    {
        const size_t J = local_atpair[iap].first;
        const size_t Q = local_atpair[iap].second;
        auto &chi0_mat= chi0_freq_q.at(J).at(Q);
        auto tmp_pi_mat= Vq_row.at(I).at(J) * chi0_mat;
        ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
        auto tmp_pi_mat2= Vq_row.at(I).at(Q) * chi0_QJ;
        omp_set_lock(&pi_lock);
        pi.at(Q)+=tmp_pi_mat;
        if(J!=Q)
            {
                pi.at(J)+=tmp_pi_mat2;
            }
        omp_unset_lock(&pi_lock);
    }
    omp_destroy_lock(&pi_lock);
    // for (auto &J_p : chi0_freq_q)
    // {
    //     const size_t J = J_p.first;
    //     for (auto &Q_p : J_p.second)
    //     {
    //         const size_t Q = Q_p.first;
    //         auto &chi0_mat = Q_p.second;
    //         pi.at(Q) += Vq_row.at(I).at(J) * chi0_mat;
    //         if (J != Q)
    //         {
    //             ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
    //             pi.at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
    //         }
    //     }
    // }
    //Pi_rowT 
    // ComplexMatrix pi_row(N_all_mu,atom_mu[I]);
    // complex<double> *pi_row_ptr=pi_row.c;
    // for(auto &Jp:pi)
    // {
    //     auto J=Jp.first;
    //     auto J_mu=atom_mu[J];
    //     const auto length=sizeof(complex<double>)* I_mu *J_mu;
    //     memcpy(pi_row_ptr, pi.at(J).c,length);
    //     pi_row_ptr+=I_mu *J_mu;
    // }
    ComplexMatrix pi_row(atom_mu[I],N_all_mu);
    for(int i=0;i!=pi_row.nr;i++)
        for(int J=0;J!=natom;J++)
            for(int j=0;j!=atom_mu[J];j++)
                pi_row(i,atom_mu_part_range[J]+j)=pi.at(J)(i,j);
    return pi_row;
}

atom_mapping<ComplexMatrix>::pair_t_old gather_vq_row_q(const int &I, const atpair_k_cplx_mat_t &coulmat, const Vector3_Order<double> &ik_vec)
{
    auto I_mu=atom_mu[I];
    atom_mapping<ComplexMatrix>::pair_t_old Vq_row;
    for(int J_tmp=0;J_tmp!=natom;J_tmp++)
    {
        auto J_mu=atom_mu[J_tmp];
        ComplexMatrix loc_vq(atom_mu[I],atom_mu[J_tmp]);
        Vq_row[I][J_tmp].create(atom_mu[I],atom_mu[J_tmp]);
       // const auto length=sizeof(complex<double>)* I_mu *J_mu;
       // complex<double> *loc_vq_ptr=loc_vq.c;
        if(I<=J_tmp)
        {
            if(Vq.count(I))
                if(Vq.at(I).count(J_tmp))
                    loc_vq=*Vq.at(I).at(J_tmp).at(ik_vec);
        }
        else
        {
            if(Vq.count(J_tmp))
                if(Vq.at(J_tmp).count(I))
                    loc_vq=transpose(*Vq.at(J_tmp).at(I).at(ik_vec), 1);
                    
        }
        mpi_comm_world_h.allreduce_ComplexMatrix(loc_vq,Vq_row[I][J_tmp]);
    }
    return Vq_row;
}

map<double, atpair_k_cplx_mat_t>
compute_Wc_freq_q(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc, const vector<std::complex<double>> &epsilon_mac_imagfreq)
{
    map<double, atpair_k_cplx_mat_t> Wc_freq_q;
    const int range_all = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    // use q-points as the outmost loop, so that square root of Coulomb will not be recalculated at each frequency point
    vector<Vector3_Order<double>> qpts;
    for ( const auto &qMuNuchi: chi0.get_chi0_q().at(chi0.tfg.get_freq_nodes()[0]))
        qpts.push_back(qMuNuchi.first);

    for (const auto &q: qpts)
    {
        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        char fn[80];

        ComplexMatrix Vq_all(range_all, range_all);
        for (const auto &Mu_NuqVq: coulmat_eps)
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for ( auto &Nu_qVq: Mu_NuqVq.second )
            {
                auto Nu = Nu_qVq.first;
                if ( 0 == Nu_qVq.second.count(q) ) continue;
                auto n_nu = atom_mu[Nu];
                for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                    {
                        Vq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) = (*Nu_qVq.second.at(q))(i_mu, i_nu);
                        Vq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) = conj((*Nu_qVq.second.at(q))(i_mu, i_nu));
                    }
            }
        }
        auto sqrtVq_all = power_hemat(Vq_all, 0.5, false, params.sqrt_coulomb_threshold);
        sprintf(fn, "sqrtVq_all_q_%d.mtx", iq);
        print_complex_matrix_mm(sqrtVq_all, fn, 1e-15);

        // truncated (cutoff) Coulomb
        ComplexMatrix Vqcut_all(range_all, range_all);
        for ( auto &Mu_NuqVq: coulmat_wc )
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for ( auto &Nu_qVq: Mu_NuqVq.second )
            {
                auto Nu = Nu_qVq.first;
                if ( 0 == Nu_qVq.second.count(q) ) continue;
                auto n_nu = atom_mu[Nu];
                for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                    {
                        Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) = (*Nu_qVq.second.at(q))(i_mu, i_nu);
                        Vqcut_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) = conj((*Nu_qVq.second.at(q))(i_mu, i_nu));
                    }
            }
        }
        auto sqrtVqcut_all = power_hemat(Vqcut_all, 0.5, true, params.sqrt_coulomb_threshold);
        // sprintf(fn, "sqrtVqcut_all_q_%d.mtx", iq);
        // print_complex_matrix_mm(sqrtVqcut_all, fn, 1e-15);
        sprintf(fn, "Vqcut_all_filtered_q_%d.mtx", iq);
        print_complex_matrix_mm(Vqcut_all, fn, 1e-15);
        // save the filtered truncated Coulomb back to the atom mapping object
        for ( auto &Mu_NuqVq: coulmat_wc )
        {
            auto Mu = Mu_NuqVq.first;
            auto n_mu = atom_mu[Mu];
            for ( auto &Nu_qVq: Mu_NuqVq.second )
            {
                auto Nu = Nu_qVq.first;
                if ( 0 == Nu_qVq.second.count(q) ) continue;
                auto n_nu = atom_mu[Nu];
                for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        (*Nu_qVq.second.at(q))(i_mu, i_nu) = Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu);
            }
        }

        ComplexMatrix chi0fq_all(range_all, range_all);
        for (const auto &freq_qMuNuchi: chi0.get_chi0_q())
        {
            auto freq = freq_qMuNuchi.first;
            auto ifreq = chi0.tfg.get_freq_index(freq);
            auto MuNuchi = freq_qMuNuchi.second.at(q);
            for (const auto &Mu_Nuchi: MuNuchi)
            {
                auto Mu = Mu_Nuchi.first;
                auto n_mu = atom_mu[Mu];
                for ( auto &Nu_chi: Mu_Nuchi.second )
                {
                    auto Nu = Nu_chi.first;
                    auto n_nu = atom_mu[Nu];
                    for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                        for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        {
                            chi0fq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) = Nu_chi.second(i_mu, i_nu);
                            chi0fq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) = conj(Nu_chi.second(i_mu, i_nu));
                        }
                }
            }
            sprintf(fn, "chi0fq_all_q_%d_freq_%d.mtx", iq, ifreq);
            print_complex_matrix_mm(chi0fq_all, fn, 1e-15);

            ComplexMatrix identity(range_all, range_all);
            identity.set_as_identity_matrix();
            auto eps_fq = identity - sqrtVq_all * chi0fq_all * sqrtVq_all;
            // sprintf(fn, "eps_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(eps_fq, fn, 1e-15);

            // invert the epsilon matrix
            power_hemat_onsite(eps_fq, -1);
            auto wc_all = sqrtVqcut_all * (eps_fq - identity) * sqrtVqcut_all;
            // sprintf(fn, "inveps_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(eps_fq, fn, 1e-15);
            // sprintf(fn, "wc_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(wc_all, fn, 1e-15);

            // save result to the atom mapping object
            for ( auto &Mu_Nuchi: MuNuchi )
            {
                auto Mu = Mu_Nuchi.first;
                auto n_mu = atom_mu[Mu];
                for ( auto &Nu_chi: Mu_Nuchi.second )
                {
                    auto Nu = Nu_chi.first;
                    auto n_nu = atom_mu[Nu];
                    shared_ptr<ComplexMatrix> wc_ptr = make_shared<ComplexMatrix>();
                    wc_ptr->create(n_mu, n_nu);
                    for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                        for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        {
                            (*wc_ptr)(i_mu, i_nu) = wc_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu);
                        }
                    Wc_freq_q[freq][Mu][Nu][q] = wc_ptr;
                }
            }
        }
    }

    return Wc_freq_q;
}

map<double, atpair_k_cplx_mat_t>
compute_Wc_freq_q_blacs(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc, const vector<std::complex<double>> &epsmac_LF_imagfreq)
{
    map<double, atpair_k_cplx_mat_t> Wc_freq_q;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const auto part_range = LIBRPA::atomic_basis_abf.get_part_range();

    if (mpi_comm_world_h.myid == 0)
    {
        cout << "Calculating Wc using ScaLAPACK" << endl;
    }
    mpi_comm_world_h.barrier();

    Array_Desc desc_nabf_nabf(LIBRPA::blacs_ctxt_world_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf = LIBRPA::get_necessary_IJ_from_block_2D_sy('U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);

    const auto atpair_ordered_local =
        dispatch_vector(tot_atpair_ordered, LIBRPA::blacs_ctxt_world_h.myid,
                        LIBRPA::blacs_ctxt_world_h.nprocs, true);
    const auto atpair_unordered_local =
        dispatch_vector(tot_atpair, LIBRPA::blacs_ctxt_world_h.myid,
                        LIBRPA::blacs_ctxt_world_h.nprocs, true);
    // LIBRPA::fout_para << "Iset Jset " << s0_s1 << endl;
    // LIBRPA::fout_para << "atpair_unordered_local of myid " << LIBRPA::blacs_ctxt_world_h.myid << " " << atpair_unordered_local << endl;

    // IJ pair of Wc to be returned
    pair<set<int>, set<int>> Iset_Jset_Wc;
    for (const auto &ap: atpair_unordered_local)
    {
        Iset_Jset_Wc.first.insert(ap.first);
        Iset_Jset_Wc.second.insert(ap.second);
    }

    vector<Vector3_Order<double>> qpts;
    for (const auto &qMuNuchi: chi0.get_chi0_q().at(chi0.tfg.get_freq_nodes()[0]))
        qpts.push_back(qMuNuchi.first);

#ifdef __USE_LIBRI
    for (const auto &q: qpts)
    {
        coul_block.zero_out();
        coulwc_block.zero_out();
        // printf("coul_block\n%s", str(coul_block).c_str());

        int iq = std::distance(klist.begin(), std::find(klist.begin(), klist.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>> couleps_libri;
            // for (const auto &Mu_NuqVq: coulmat_eps)
            // {
            //     const auto Mu = Mu_NuqVq.first;
            //     const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
            //     for (const auto &Nu_qVq: Mu_NuqVq.second)
            //     {
            //         const auto Nu = Nu_qVq.first;
            //         const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
            //         if (Nu_qVq.second.count(q) == 0) continue;
            //         const auto Vq = Nu_qVq.second.at(q);
            //         // NOTE: consume extra memoery when creating valarray from shared_ptr
            //         std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
            //         auto pvq = std::make_shared<std::valarray<complex<double>>>();
            //         *pvq = Vq_va;
            //         couleps_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
            //     }
            // }
            for (const auto &Mu_Nu: atpair_unordered_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                LIBRPA::fout_para << "myid " << LIBRPA::blacs_ctxt_world_h.myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_eps.count(Mu) == 0 ||
                    coulmat_eps.at(Mu).count(Nu) == 0 ||
                    coulmat_eps.at(Mu).at(Nu).count(q) == 0) continue;
                const auto &Vq = coulmat_eps.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            // LIBRPA::fout_para << "Couleps_libri" << endl << couleps_libri;
            // if (couleps_libri.size() == 0)
            //     throw std::logic_error("data at q-point not found in coulmat_eps");

            // perform communication
            // cout << "Iset&Jset" << Iset_Jset << endl;
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(LIBRPA::mpi_comm_world_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            // LIBRPA::fout_para << "IJq_coul" << endl << IJq_coul;

            for (const auto &IJ: set_IJ_nabf_nabf)
            {
                const auto &I = IJ.first;
                const auto &J = IJ.second;
                // cout << IJq_coul.at(I).at({J, qa});
                collect_block_from_IJ_storage_syhe(
                    coul_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
                    IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
                // printf("myid %d I %d J %d nr %d nc %d\n%s",
                //        LIBRPA::blacs_ctxt_world_h.myid, I, J,
                //        coul_block.nr(), coul_block.nc(),
                //        str(coul_block).c_str());
            }
        }
        char fn[100];
        // sprintf(fn, "couleps_iq_%d.mtx", iq);
        // print_matrix_mm_file_parallel(fn, coul_block, desc_nabf_nabf);
        // LIBRPA::fout_para << str(coul_block);
        // printf("coul_block\n%s", str(coul_block).c_str());
        size_t n_singular;
        vec<double> eigenvalues(n_abf);
        auto sqrtveig_blacs = power_hemat_blacs(coul_block, desc_nabf_nabf, coul_eigen_block, desc_nabf_nabf, n_singular, eigenvalues.c, 0.5, params.sqrt_coulomb_threshold);
        // printf("nabf %d nsingu %lu\n", n_abf, n_singular);
        // release sqrtv when the q-point is not Gamma, or macroscopic dielectric constant at imaginary frequency is not prepared
        if (epsmac_LF_imagfreq.empty() || !is_gamma_point(q))
            sqrtveig_blacs.clear();
        const size_t n_nonsingular = n_abf - n_singular;

        // collect the block elements of truncated coulomb matrices
        {
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>> couleps_libri;
            // for (const auto &Mu_NuqVq: coulmat_wc)
            // {
            //     const auto Mu = Mu_NuqVq.first;
            //     const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
            //     for (const auto &Nu_qVq: Mu_NuqVq.second)
            //     {
            //         const auto Nu = Nu_qVq.first;
            //         const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
            //         if (Nu_qVq.second.count(q) == 0) continue;
            //         const auto Vq = Nu_qVq.second.at(q);
            //         // NOTE: will it consume extra memoery when creating valarray from shared_ptr?
            //         std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
            //         auto pvq = std::make_shared<std::valarray<complex<double>>>();
            //         *pvq = Vq_va;
            //         couleps_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
            //     }
            // }
            for (const auto &Mu_Nu: atpair_unordered_local)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                LIBRPA::fout_para << "myid " << LIBRPA::blacs_ctxt_world_h.myid << "Mu " << Mu << " Nu " << Nu << endl;
                if (coulmat_wc.count(Mu) == 0 ||
                    coulmat_wc.at(Mu).count(Nu) == 0 ||
                    coulmat_wc.at(Mu).at(Nu).count(q) == 0) continue;
                const auto &Vq = coulmat_wc.at(Mu).at(Nu).at(q);
                const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            // if (couleps_libri.size() == 0)
            //     throw std::logic_error("data at q-point not found in coulmat_wc");

            // perform communication
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(LIBRPA::mpi_comm_world_h.comm, couleps_libri, s0_s1.first, s0_s1.second);

            for (const auto &IJ: set_IJ_nabf_nabf)
            {
                const auto &I = IJ.first;
                const auto &J = IJ.second;
                collect_block_from_IJ_storage_syhe(
                    coulwc_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
                    IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            }
            // WARN: check whether one should pass back the filtered coulwc_block
            // sprintf(fn, "coulwc_iq_%d.mtx", iq);
            // print_matrix_mm_file_parallel(fn, coulwc_block, desc_nabf_nabf);
            power_hemat_blacs(coulwc_block, desc_nabf_nabf, coul_eigen_block, desc_nabf_nabf, n_singular, eigenvalues.c, 0.5, params.sqrt_coulomb_threshold);
        }

        for (const auto &freq: chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            chi0_block.zero_out();
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>> chi0_libri;
                const auto &chi0_wq = chi0.get_chi0_q().at(freq).at(q);
                // for (const auto &Mu_Nu: atpair_unordered_local)
                // {
                //     const auto Mu = Mu_Nu.first;
                //     const auto Nu = Mu_Nu.second;
                //     // LIBRPA::fout_para << "myid " << LIBRPA::blacs_ctxt_world_h.myid << "Mu " << Mu << " Nu " << Nu << endl;
                //     if (chi0_wq.count(Mu) == 0 ||
                //         chi0_wq.at(Mu).count(Nu) == 0 ) continue;
                //     const auto &chi = chi0_wq.at(Mu).at(Nu);
                //     const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
                //     const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
                //     std::valarray<complex<double>> chi_va(chi.c, chi.size);
                //     auto pvq = std::make_shared<std::valarray<complex<double>>>();
                //     *pvq = chi_va;
                //     chi0_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
                // }
                for (const auto &M_Nchi: chi0_wq)
                {
                    const auto &M = M_Nchi.first;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    for (const auto &N_chi: M_Nchi.second)
                    {
                        const auto &N = N_chi.first;
                        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                        const auto &chi = N_chi.second;
                        std::valarray<complex<double>> chi_va(chi.c, chi.size);
                        auto pchi = std::make_shared<std::valarray<complex<double>>>();
                        *pchi = chi_va;
                        chi0_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                    }
                }
                // LIBRPA::fout_para << "chi0_libri" << endl << chi0_libri;
                const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(LIBRPA::mpi_comm_world_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // LIBRPA::fout_para << "IJq_chi0" << endl << IJq_chi0;
                for (const auto &IJ: set_IJ_nabf_nabf)
                {
                    const auto &I = IJ.first;
                    const auto &J = IJ.second;
                    collect_block_from_IJ_storage_syhe(
                        chi0_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, IJ.first,
                        IJ.second, true, CONE, IJq_chi0.at(I).at({J, qa}).ptr(), MAJOR::ROW);
                }
                // sprintf(fn, "chi_ifreq_%d_iq_%d.mtx", ifreq, iq);
                // print_matrix_mm_file_parallel(fn, chi0_block, desc_nabf_nabf);
            }
            // for Gamma point, overwrite the head term
            if (epsmac_LF_imagfreq.size() > 0 && is_gamma_point(q))
            {
                // rotate to Coulomb-eigenvector basis
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_nonsingular, n_abf, 1.0,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                        sqrtveig_blacs.ptr(), 1, n_singular+1, desc_nabf_nabf.desc, 0.0,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
                ScalapackConnector::pgemm_f('C', 'N', n_nonsingular, n_nonsingular, n_abf, 1.0,
                        sqrtveig_blacs.ptr(), 1, n_singular+1, desc_nabf_nabf.desc,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
                const int ilo = desc_nabf_nabf.indx_g2l_r(n_nonsingular);
                const int jlo = desc_nabf_nabf.indx_g2l_c(n_nonsingular);
                if (ilo >= 0 && jlo >= 0)
                    chi0_block(ilo, jlo) = - epsmac_LF_imagfreq[ifreq];
                // rotate back to ABF
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_nonsingular, n_nonsingular, 1.0,
                        coul_eigen_block.ptr(), 1, n_singular+1, desc_nabf_nabf.desc,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
                ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, n_nonsingular, 1.0,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                        coul_eigen_block.ptr(), 1, n_singular+1, desc_nabf_nabf.desc, 0.0,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
            }
            else
            {
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0,
                        coul_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0,
                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                        coul_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                        chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
            }
            // now chi0_block is actually v1/2 chi v1/2
            chi0_block *= -1.0;
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                if (ilo < 0) continue;
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (jlo < 0) continue;
                chi0_block(ilo, jlo) += 1.0;
            }
            // now chi0_block is actually the dielectric matrix
            // perform inversion
            invert_scalapack(chi0_block, desc_nabf_nabf);
            // subtract 1 from diagonal
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf.indx_g2l_r(i);
                if (ilo < 0) continue;
                const int jlo = desc_nabf_nabf.indx_g2l_c(i);
                if (jlo < 0) continue;
                chi0_block(ilo, jlo) -= 1.0;
            }
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0,
                    coulwc_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                    chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                    coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
            ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0,
                    coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                    coulwc_block.ptr(), 1, 1, desc_nabf_nabf.desc, 0.0,
                    chi0_block.ptr(), 1, 1, desc_nabf_nabf.desc);
            // printf("chi0_block\n%s", str(chi0_block).c_str());
            // now chi0_block is the screened Coulomb interaction Wc (i.e. W-V)
            map<int, map<int, matrix_m<complex<double>>>> Wc_MNmap;
            map_block_to_IJ_storage(Wc_MNmap, LIBRPA::atomic_basis_abf,
                                    LIBRPA::atomic_basis_abf, chi0_block,
                                    desc_nabf_nabf, MAJOR::ROW);
            // cout << Wc_MNmap;
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>> Wc_libri;
                for (const auto &M_NWc: Wc_MNmap)
                {
                    const auto &M = M_NWc.first;
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    for (const auto &N_Wc: M_NWc.second)
                    {
                        const auto &N = N_Wc.first;
                        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                        const auto &Wc = N_Wc.second;
                        // std::valarray<complex<double>> Wc_va(Wc.ptr(), Wc.size());
                        // auto pWc = std::make_shared<std::valarray<complex<double>>>();
                        // *pWc = Wc_va;
                        Wc_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, Wc.sptr());
                    }
                }
                // cout << Wc_libri;
                const auto IJq_Wc = RI::Communicate_Tensors_Map_Judge::comm_map2_first(LIBRPA::mpi_comm_world_h.comm, Wc_libri, Iset_Jset_Wc.first, Iset_Jset_Wc.second);
                // parse collected to 
                for (const auto &MN: atpair_unordered_local)
                {
                    const auto &M = MN.first;
                    const auto &N = MN.second;
                    shared_ptr<ComplexMatrix> wc_ptr = make_shared<ComplexMatrix>();
                    const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(M);
                    const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(N);
                    wc_ptr->create(n_mu, n_nu);
                    for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                        for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        {
                            (*wc_ptr)(i_mu, i_nu) = IJq_Wc.at(M).at({N, qa})(i_mu, i_nu);
                        }
                    Wc_freq_q[freq][M][N][q] = wc_ptr;
                }
                // for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                //     for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                //     {
                //     }
            }
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    return Wc_freq_q;
}

map<double, atpair_R_cplx_mat_t>
CT_FT_Wc_freq_q(const map<double, atpair_k_cplx_mat_t> &Wc_freq_q,
                const TFGrids &tfg, vector<Vector3_Order<int>> Rlist)
{
    map<double, atpair_R_cplx_mat_t> Wc_tau_R;
    if (!tfg.has_time_grids())
        throw logic_error("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();
    for (auto R: Rlist)
    {
        for (int itau = 0; itau != ngrids; itau++)
        {
            auto tau = tfg.get_time_nodes()[itau];
            for (int ifreq = 0; ifreq != ngrids; ifreq++)
            {
                auto freq = tfg.get_freq_nodes()[ifreq];
                auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
                if (Wc_freq_q.count(freq))
                {
                    for (const auto &Mu_NuqWc: Wc_freq_q.at(freq))
                    {
                        const auto Mu = Mu_NuqWc.first;
                        const int n_mu = atom_mu[Mu];
                        for (const auto &Nu_qWc: Mu_NuqWc.second)
                        {
                            const auto Nu = Nu_qWc.first;
                            const int n_nu = atom_mu[Nu];
                            if(!(Wc_tau_R.count(tau) &&
                                 Wc_tau_R.at(tau).count(Mu) &&
                                 Wc_tau_R.at(tau).at(Mu).count(Nu) &&
                                 Wc_tau_R.at(tau).at(Mu).at(Nu).count(R)))
                            {
                                Wc_tau_R[tau][Mu][Nu][R] = make_shared<ComplexMatrix>();
                                Wc_tau_R[tau][Mu][Nu][R]->create(n_mu, n_nu);
                            }
                            auto & WtR = Wc_tau_R[tau][Mu][Nu][R];
                            for (const auto &q_Wc: Nu_qWc.second)
                            {
                                auto q = q_Wc.first;
                                for (auto q_bz: map_irk_ks[q])
                                {
                                    double ang = - q_bz * (R * latvec) * TWO_PI;
                                    complex<double> kphase = complex<double>(cos(ang), sin(ang));
                                    complex<double> weight = kphase * f2t;
                                    if (q == q_bz)
                                        *WtR += (*q_Wc.second) * weight;
                                    else
                                        *WtR += conj(*q_Wc.second) * weight;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // myz debug: check the imaginary part of the matrix
    // NOTE: if G(R) is real, is W(R) real as well?
    for (const auto & tau_MuNuRWc: Wc_tau_R)
    {
        char fn[80];
        auto tau = tau_MuNuRWc.first;
        auto itau = tfg.get_time_index(tau);
        for (const auto & Mu_NuRWc: tau_MuNuRWc.second)
        {
            auto Mu = Mu_NuRWc.first;
            // const int n_mu = atom_mu[Mu];
            for (const auto & Nu_RWc: Mu_NuRWc.second)
            {
                auto Nu = Nu_RWc.first;
                // const int n_nu = atom_mu[Nu];
                for (const auto & R_Wc: Nu_RWc.second)
                {
                    auto R = R_Wc.first;
                    auto Wc = R_Wc.second;
                    auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                    auto iR = std::distance(Rlist.cbegin(), iteR);
                    sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_itau_%d.mtx", Mu, Nu, iR, itau);
                    print_complex_matrix_mm(*Wc, fn);
                }
            }
        }
    }
    // end myz debug
    return Wc_tau_R;
}
