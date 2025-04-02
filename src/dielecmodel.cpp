#include "dielecmodel.h"

#include <cmath>
#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif

const int DoubleHavriliakNegami::d_npar = 8;

const std::function<double(double, const std::vector<double> &)>
    DoubleHavriliakNegami::func_imfreq = [](double u, const std::vector<double> &pars)
{
    return 1.0 + (pars[0] - 1.0) / std::pow(1.0 + std::pow(u * pars[3], pars[1]), pars[2]) +
           (pars[4] - 1.0) / std::pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
};

const std::function<void(std::vector<double> &, double, const std::vector<double> &)>
    DoubleHavriliakNegami::grad_imfreq =
        [](std::vector<double> &grads, double u, const std::vector<double> &pars)
{
    using std::pow;
    using std::log;
    grads[0] = 1.0 / pow(1.0 + pow(u * pars[3], pars[1]), pars[2]);
    grads[1] = (pars[0] - 1.0) * (-pars[2]) / pow(1.0 + pow(u * pars[3], pars[1]), pars[2] + 1) *
               log(u * pars[3]) * pow(u * pars[3], pars[1]);
    grads[2] = (1.0 - pars[0]) * log(1.0 + pow(u * pars[3], pars[1])) /
               pow(1.0 + pow(u * pars[3], pars[1]), pars[2]);
    grads[3] = (pars[0] - 1.0) * (-pars[2]) / pow(1.0 + pow(u * pars[3], pars[1]), pars[2] + 1) *
               pars[1] / pars[3] * pow(u * pars[3], pars[1]);
    grads[4] = 1.0 / pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
    grads[5] = (pars[4] - 1.0) * (-pars[6]) / pow(1.0 + pow(u * pars[7], pars[5]), pars[6] + 1) *
               log(u * pars[7]) * pow(u * pars[7], pars[5]);
    grads[6] = (1.0 - pars[4]) * log(1.0 + pow(u * pars[7], pars[5])) /
               pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
    grads[7] = (pars[4] - 1.0) * (-pars[6]) / pow(1.0 + pow(u * pars[7], pars[5]), pars[6] + 1) *
               pars[5] / pars[7] * pow(u * pars[7], pars[5]);
};

void diele_func::cal_head()
{
    int nk = this->meanfield_df.get_n_kpoints();
    int nbands = this->meanfield_df.get_n_bands();
    int nspin = this->meanfield_df.get_n_spins();  //! spin = 1 only
    auto &wg = this->meanfield_df.get_weight()[nspin - 1];
    auto &eigenvalues = this->meanfield_df.get_eigenvals()[nspin - 1];
    auto &velocity = this->meanfield_df.get_velocity()[nspin - 1];
    std::complex<double> tmp;
    int nocc = 0;

    double dielectric_unit = cal_factor("head");
    for (int i = 0; i != wg.size; i++)
    {
        if (wg.c[i] == 0.)
        {
            nocc = i;
            break;
        }
    }
    for (int ik = 0; ik != nk; ik++)
    {
        for (int iocc = 0; iocc != nocc; iocc++)
        {
            for (int iunocc = nocc; iunocc != nbands; iunocc++)
            {
                double egap = (eigenvalues(ik, iocc) - eigenvalues(ik, iunocc)) * HA2EV;
                for (int alpha = 0; alpha != 3; alpha++)
                {
                    for (int beta = 0; beta != 3; beta++)
                    {
                        for (int iomega = 0; iomega != this->omega.size(); iomega++)
                        {
                            double omega_ev = this->omega[iomega] * HA2EV;
                            tmp = 2.0 * velocity[ik][alpha](iunocc, iocc) *
                                  velocity[ik][beta](iocc, iunocc) /
                                  (egap * egap + omega_ev * omega_ev) / egap;
                            this->head.at(alpha).at(beta).at(iomega) -= tmp;
                        }
                    }
                }
            }
        }
    }
    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int beta = 0; beta != 3; beta++)
        {
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                if (nspin == 1)
                {
                    this->head.at(alpha).at(beta).at(iomega) *= dielectric_unit * 2;
                }
                else if (nspin == 4)
                    this->head.at(alpha).at(beta).at(iomega) *= dielectric_unit;
                if (alpha == beta)
                {
                    this->head.at(alpha).at(beta).at(iomega) += complex<double>(1.0, 0.0);
                }
            }
        }
    }
    std::cout << "* Success: calculate head term.\n";
};

double diele_func::cal_factor(string name)
{
    const double h_divide_e2 = 25812.80745;
    const double epsilon0 = 8.854187817e-12;
    const double hbar = 1.05457182e-34;
    const double eV = 1.60217662e-19;
    double dielectric_unit;
    //! Bohr to A
    const double primitive_cell_volume = latvec.Det() * BOHR2ANG * BOHR2ANG * BOHR2ANG;
    // latvec.print();
    if (name == "head")
        dielectric_unit = 2 * TWO_PI / primitive_cell_volume / this->meanfield_df.get_n_kpoints() *
                          1.0e30 / epsilon0 / eV;
    else if (name == "wing")
        dielectric_unit = 2 * sqrt(2 * TWO_PI / primitive_cell_volume * 1.0e30) /
                          this->meanfield_df.get_n_kpoints() / epsilon0 / eV;
    else
        throw std::logic_error("Unsupported value for head/wing factor");
    return dielectric_unit;
};

void diele_func::init_headwing()
{
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    get_Xv();
    head.clear();
    wing.clear();
    head.resize(3);
    wing.resize(3);
    for (int alpha = 0; alpha < 3; alpha++)
    {
        head[alpha].resize(3);
        wing[alpha].resize(n_nonsingular - 1);
        for (int beta = 0; beta < 3; beta++)
        {
            head[alpha][beta].resize(this->omega.size());
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                this->head.at(alpha).at(beta).at(iomega) = complex<double>(0.0, 0.0);
            }
        }
        for (int lambda = 0; lambda < n_nonsingular - 1; lambda++)
        {
            wing[alpha][lambda].resize(this->omega.size());
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                this->wing.at(alpha).at(lambda).at(iomega) = complex<double>(0.0, 0.0);
            }
        }
    }
};

void diele_func::test_head()
{
    std::cout << "BEGIN test head !!!!!!!!!!" << std::endl;
    for (int iomega = 0; iomega != this->omega.size(); iomega++)
    {
        std::complex<double> df = 0;
        for (int alpha = 0; alpha != 3; alpha++)
        {
            df += this->head.at(alpha).at(alpha).at(iomega);
        }
        std::cout << this->omega[iomega] << " " << df.real() / 3.0 << " " << df.imag() / 3.0
                  << std::endl;
    }
    std::cout << "END test head !!!!!!!!!!" << std::endl;
    // std::exit(0);
};

void diele_func::cal_wing()
{
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    int n_lambda = this->n_nonsingular - 1;
    int nk = kfrac_band.size();
    auto &velocity = this->meanfield_df.get_velocity();
    auto &eigenvalues = this->meanfield_df.get_eigenvals();
    auto &wg = this->meanfield_df.get_weight()[n_spin - 1];
    int nocc = 0;
    for (int i = 0; i != wg.size; i++)
    {
        if (wg.c[i] == 0.)
        {
            nocc = i;
            break;
        }
    }
    init_Cs();
    FT_R2k();
    Cs_ij2mn();
#pragma omp parallel for schedule(dynamic) collapse(3)
    for (int iomega = 0; iomega != this->omega.size(); iomega++)
    {
        for (int alpha = 0; alpha != 3; alpha++)
        {
            for (int il = 0; il != n_lambda; il++)
            {
                for (int mu = 0; mu != n_abf; mu++)
                {
                    std::complex<double> tmp = 0.0;
                    double omega_ev = this->omega[iomega] * HA2EV;
                    for (int ik = 0; ik != nk; ik++)
                    {
                        for (int ispin = 0; ispin != n_spin; ispin++)
                        {
                            for (int iocc = 0; iocc != nocc; iocc++)
                            {
                                for (int iunocc = nocc; iunocc != n_states; iunocc++)
                                {
                                    double egap = (eigenvalues[ispin](ik, iunocc) -
                                                   eigenvalues[ispin](ik, iocc)) *
                                                  HA2EV;

                                    tmp += conj(this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                                velocity[ispin][ik][alpha](iunocc, iocc)) /
                                           (omega_ev * omega_ev + egap * egap);
                                }
                            }
                        }
                    }
                    this->wing.at(alpha).at(il).at(iomega) += this->Coul_vector.at(il).at(mu) * tmp;
                }
            }
        }
    }
    double dielectric_unit = cal_factor("wing");

    //---------------test-----------------
    // std::cout << "factor: " << dielectric_unit << std::endl;
    // for (int il = 0; il != n_lambda; il++)
    // {
    //     std::cout << "Vq eigenvalue: " << il << "," << this->Coul_value.at(il) << std::endl;
    //     for (int mu = 0; mu != n_abf; mu++)
    //     {
    //         std::cout << "Vq eigenvecotr: " << mu << this->Coul_vector.at(il).at(mu) <<
    //         std::endl;
    //     }
    // }
    //---------------test-----------------
    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int il = 0; il != n_lambda; il++)
        {
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                if (n_spin == 1)
                {
                    this->wing.at(alpha).at(il).at(iomega) *=
                        -dielectric_unit * sqrt(this->Coul_value.at(il)) * 2;
                }
                else if (n_spin == 4)
                    this->wing.at(alpha).at(il).at(iomega) *=
                        -dielectric_unit * sqrt(this->Coul_value.at(il));
            }
        }
    }
    std::cout << "* Success: calculate wing term.\n";
};

void diele_func::init_Cs()
{
    for (auto k_frac : this->kfrac_band)
    {
        const std::array<int, 3> k_array = {k_frac.x, k_frac.y, k_frac.z};
        for (const auto &outer : Cs_data.data_libri)
        {
            int I = outer.first;
            for (const auto &inner : outer.second)
            {
                std::pair<int, std::array<int, 3UL>> pair = inner.first;
                int J = pair.first;
                int n_mu_I = inner.second.shape[0];
                int n_ao_I = inner.second.shape[1];
                int n_ao_J = inner.second.shape[2];
                size_t total = n_ao_I * n_ao_J * n_mu_I;
                std::complex<double> *Cs_in = new std::complex<double>[total]();
                matrix_m<std::complex<double>> mat(n_ao_I * n_ao_J, n_mu_I, Cs_in, MAJOR::ROW,
                                                   MAJOR::COL);
                const std::initializer_list<std::size_t> shape{static_cast<std::size_t>(n_mu_I),
                                                               static_cast<std::size_t>(n_ao_I),
                                                               static_cast<std::size_t>(n_ao_J)};
                Ctri_ij.data_libri[I][{J, k_array}] =
                    RI::Tensor<std::complex<double>>(shape, mat.dataobj.data);
                delete[] Cs_in;
            }
        }
    }
    int nk = this->kfrac_band.size();
    int nbands = this->meanfield_df.get_n_bands();
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const int n_atom = Ctri_ij.data_libri.size();
    for (int ik = 0; ik != nk; ik++)
    {
        for (int m = 0; m != nbands; m++)
        {
            for (int n = 0; n != nbands; n++)
            {
                for (const auto &outer : Ctri_ij.data_libri)
                {
                    int Mu = outer.first;
                    for (const auto &inner : outer.second)
                    {
                        std::pair<int, std::array<int, 3UL>> pair = inner.first;
                        int J = pair.first;
                        int n_mu_I = inner.second.shape[0];
                        this->Ctri_mn.resize(n_mu_I * n_atom);
                        for (int mu = 0; mu != n_mu_I; mu++)
                        {
                            this->Ctri_mn.at(n_mu_I * Mu + mu).resize(nbands);
                            this->Ctri_mn.at(n_mu_I * Mu + mu).at(m).resize(nbands);
                            this->Ctri_mn.at(n_mu_I * Mu + mu)
                                .at(m)
                                .at(n)
                                .insert(std::make_pair(kfrac_band[ik], 0.0));
                        }
                    }
                }
            }
        }
    }
};

void diele_func::FT_R2k()
{
// Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
// auto Rlist = construct_R_grid(period);
#pragma omp parallel for schedule(dynamic)
    for (auto k_frac : this->kfrac_band)
    {
        const std::array<int, 3> k_array = {k_frac.x, k_frac.y, k_frac.z};
        for (const auto &outer : Cs_data.data_libri)
        {
            int I = outer.first;
            for (const auto &inner : outer.second)
            {
                std::pair<int, std::array<int, 3UL>> pair = inner.first;
                int J = pair.first;
                std::array<int, 3UL> Ra = pair.second;
                Vector3_Order<double> R = {Ra[0], Ra[1], Ra[2]};
                double ang = k_frac * R * TWO_PI;
                complex<double> kphase = complex<double>(cos(ang), sin(ang));
                int n_mu_I = inner.second.shape[0];
                int n_ao_I = inner.second.shape[1];
                int n_ao_J = inner.second.shape[2];
                for (int mu = 0; mu != n_mu_I; mu++)
                {
                    for (int i = 0; i != n_ao_I; i++)
                    {
                        for (int j = 0; j != n_ao_J; j++)
                        {
                            this->Ctri_ij.data_libri[I][{J, k_array}](mu, i, j) +=
                                kphase * inner.second(mu, i, j);
                        }
                    }
                }
            }
        }
    }
    std::cout << "* Success: Fourier transform from Cs(R) to Cs(k).\n";
};

void diele_func::Cs_ij2mn()
{
    using LIBRPA::atomic_basis_wfc;
    int nspin = this->meanfield_df.get_n_spins();  //! spin = 1 only
    int nk = this->kfrac_band.size();
    int nbands = this->meanfield_df.get_n_bands();
    std::complex<double> term1 = 0.0;
    std::complex<double> term2 = 0.0;
    for (int ispin = 0; ispin != nspin; ispin++)
    {
#pragma omp parallel for schedule(dynamic) collapse(3)
        for (int ik = 0; ik != nk; ik++)
        {
            for (int m = 0; m != nbands; m++)
            {
                for (int n = 0; n != nbands; n++)
                {
                    for (const auto &outer : Ctri_ij.data_libri)
                    {
                        int Mu = outer.first;
                        for (const auto &inner : outer.second)
                        {
                            std::pair<int, std::array<int, 3UL>> pair = inner.first;
                            int J = pair.first;
                            int n_mu_I = inner.second.shape[0];
                            int n_ao_I = inner.second.shape[1];
                            int n_ao_J = inner.second.shape[2];
                            for (int i = 0; i != n_ao_I; i++)
                            {
                                for (int j = 0; j != n_ao_J; j++)
                                {
                                    for (int mu = 0; mu != n_mu_I; mu++)
                                    {
                                        term1 = conj(meanfield_df.get_eigenvectors()[ispin][ik](
                                                    m, atomic_basis_wfc.get_global_index(Mu, i))) *
                                                inner.second(mu, i, j) *
                                                meanfield_df.get_eigenvectors()[ispin][ik](
                                                    n, atomic_basis_wfc.get_global_index(J, j));
                                        term2 = meanfield_df.get_eigenvectors()[ispin][ik](
                                                    n, atomic_basis_wfc.get_global_index(Mu, i)) *
                                                conj(inner.second(mu, i, j)) *
                                                conj(meanfield_df.get_eigenvectors()[ispin][ik](
                                                    m, atomic_basis_wfc.get_global_index(J, j)));
                                        this->Ctri_mn.at(n_mu_I * Mu + mu)
                                            .at(m)
                                            .at(n)
                                            .at(kfrac_band[ik]) += term1 + term2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "* Success: transform of Cs^mu_ij(k) to Cs^mu_mn(k).\n";
};

void diele_func::get_Xv()
{
    this->Coul_vector.clear();
    this->Coul_value.clear();
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    const complex<double> CONE{1.0, 0.0};
    std::array<double, 3> qa = {0.0, 0.0, 0.0};
    Vector3_Order<double> q = {0.0, 0.0, 0.0};
    size_t n_singular;
    vec<double> eigenvalues(n_abf);
    using LIBRPA::Array_Desc;
    using LIBRPA::envs::blacs_ctxt_global_h;
    using LIBRPA::envs::mpi_comm_global_h;

    mpi_comm_global_h.barrier();

    Array_Desc desc_nabf_nabf(blacs_ctxt_global_h);
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf =
        LIBRPA::get_necessary_IJ_from_block_2D_sy('U', LIBRPA::atomic_basis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    coulwc_block.zero_out();
    std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
        couleps_libri;
    const auto atpair_local = dispatch_upper_trangular_tasks(
        natom, blacs_ctxt_global_h.myid, blacs_ctxt_global_h.nprows, blacs_ctxt_global_h.npcols,
        blacs_ctxt_global_h.myprow, blacs_ctxt_global_h.mypcol);
    for (const auto &Mu_Nu : atpair_local)
    {
        const auto Mu = Mu_Nu.first;
        const auto Nu = Mu_Nu.second;
        // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
        if (Vq_cut.count(Mu) == 0 || Vq_cut.at(Mu).count(Nu) == 0 ||
            Vq_cut.at(Mu).at(Nu).count(q) == 0)
            continue;
        const auto &Vq = Vq_cut.at(Mu).at(Nu).at(q);
        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
        std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
        auto pvq = std::make_shared<std::valarray<complex<double>>>();
        *pvq = Vq_va;
        couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
    }
    const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
    collect_block_from_ALL_IJ_Tensor(coulwc_block, desc_nabf_nabf, LIBRPA::atomic_basis_abf, qa,
                                     true, CONE, IJq_coul, MAJOR::ROW);
    power_hemat_blacs(coulwc_block, desc_nabf_nabf, coul_eigen_block, desc_nabf_nabf, n_singular,
                      eigenvalues.c, 1.0, Params::vq_threshold);
    this->n_nonsingular = n_abf - n_singular;
    for (int iv = 0; iv != n_nonsingular - 1; iv++)
    {
        this->Coul_value.push_back(eigenvalues.c[iv + 1]);
        std::vector<std::complex<double>> newRow;
        for (int jabf = 0; jabf != n_abf; jabf++)
        {
            newRow.push_back(coul_eigen_block.dataobj[(iv + 1) * n_abf + jabf]);
        }
        this->Coul_vector.push_back(newRow);
    }
    std::reverse(this->Coul_value.begin(), this->Coul_value.end());
    std::reverse(this->Coul_vector.begin(), this->Coul_vector.end());
    std::cout << "* Success: diagonalize Coulomb matrix in the ABFs repre.\n";
};

void diele_func::test_wing()
{
    std::cout << "BEGIN test wing !!!!!!!!!!" << std::endl;
    for (int iomega = 0; iomega != this->omega.size(); iomega++)
    {
        std::complex<double> df = 0;
        // z direction and the last lambda
        df += this->wing.at(2).at(n_nonsingular - 2).at(iomega);

        std::cout << this->omega[iomega] << " " << df.real() << " " << df.imag() << std::endl;
    }
    std::cout << "END test wing !!!!!!!!!!" << std::endl;
    std::exit(0);
};