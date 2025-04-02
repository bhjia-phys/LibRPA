#include "dielecmodel.h"

#include <cmath>
#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif
using LIBRPA::utils::lib_printf;

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
                double egap = (eigenvalues(ik, iocc) - eigenvalues(ik, iunocc));  // * HA2EV;
                for (int alpha = 0; alpha != 3; alpha++)
                {
                    for (int beta = 0; beta != 3; beta++)
                    {
                        for (int iomega = 0; iomega != this->omega.size(); iomega++)
                        {
                            double omega_ev = this->omega[iomega];  // * HA2EV;
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
    const double primitive_cell_volume = latvec.Det();  //* BOHR2ANG * BOHR2ANG * BOHR2ANG;
    // latvec.print();
    if (name == "head")
    {
        // abacus
        /*dielectric_unit = TWO_PI * hbar / h_divide_e2 / primitive_cell_volume /
                          this->meanfield_df.get_n_kpoints() * 1.0e30 / epsilon0 / eV;*/
        // aims
        dielectric_unit = 2 * TWO_PI / primitive_cell_volume / this->meanfield_df.get_n_kpoints();
    }
    else if (name == "wing")
    {
        // abacus
        /* dielectric_unit = TWO_PI * hbar / h_divide_e2 * sqrt(2 * TWO_PI / primitive_cell_volume)
         * 1.0e15 / this->meanfield_df.get_n_kpoints() / epsilon0 / TWO_PI / eV; */
        // aims
        dielectric_unit = 2 * sqrt(2 * TWO_PI / primitive_cell_volume) /
                          this->meanfield_df.get_n_kpoints();  // bohr
    }
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
    wing_mu.clear();
    head.resize(3);
    wing.resize(3);
    wing_mu.resize(3);
    for (int alpha = 0; alpha < 3; alpha++)
    {
        head[alpha].resize(3);
        wing[alpha].resize(n_nonsingular - 1);
        wing_mu[alpha].resize(n_abf);
        for (int beta = 0; beta < 3; beta++)
        {
            head[alpha][beta].resize(this->omega.size());
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                this->head.at(alpha).at(beta).at(iomega) = complex<double>(0.0, 0.0);
            }
        }
        for (int mu = 0; mu < n_abf; mu++)
        {
            wing_mu[alpha][mu].resize(this->omega.size());
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                this->wing_mu.at(alpha).at(mu).at(iomega) = complex<double>(0.0, 0.0);
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
            // std::cout << alpha << ", " << this->head.at(alpha).at(alpha).at(iomega) << std::endl;
        }
        std::cout << this->omega[iomega] << " " << df.real() / 3.0 << " " << df.imag() / 3.0
                  << std::endl;
    }
    std::cout << "END test head !!!!!!!!!!" << std::endl;
    // std::exit(0);
};

void diele_func::cal_wing()
{
    int n_lambda = this->n_nonsingular - 1;
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;

    init_Cs();
    FT_R2k();
    Cs_ij2mn();

#pragma omp parallel for schedule(dynamic) collapse(3)
    for (int iomega = 0; iomega != this->omega.size(); iomega++)
    {
        for (int alpha = 0; alpha != 3; alpha++)
        {
            for (int mu = 0; mu != n_abf; mu++)
            {
                this->wing_mu.at(alpha).at(mu).at(iomega) = compute_wing(alpha, iomega, mu);
            }
        }
    }
    double dielectric_unit = cal_factor("wing");

    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int mu = 0; mu != n_abf; mu++)
        {
            for (int iomega = 0; iomega != this->omega.size(); iomega++)
            {
                if (n_spin == 1)
                {
                    this->wing_mu.at(alpha).at(mu).at(iomega) *= -dielectric_unit * 2.0;
                }
                else if (n_spin == 4)
                    this->wing_mu.at(alpha).at(mu).at(iomega) *= -dielectric_unit;
            }
        }
    }
    tranform_mu_to_lambda();
    std::cout << "* Success: calculate wing term.\n";
};

void diele_func::tranform_mu_to_lambda()
{
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    int n_lambda = this->n_nonsingular - 1;
    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int iomega = 0; iomega != this->omega.size(); iomega++)
        {
            for (int lambda = 0; lambda != n_lambda; lambda++)
            {
                for (int mu = 0; mu != n_abf; mu++)
                {
                    this->wing.at(alpha).at(lambda).at(iomega) +=
                        conj(this->Coul_vector.at(lambda).at(mu)) *
                        wing_mu.at(alpha).at(mu).at(iomega);
                }
                this->wing.at(alpha).at(lambda).at(iomega) *= sqrt(this->Coul_value.at(lambda));
            }
        }
    }
};

std::complex<double> diele_func::compute_wing(int alpha, int iomega, int mu)
{
    const int nk = this->meanfield_df.get_n_kpoints();
    const int nbands = this->meanfield_df.get_n_bands();
    auto &wg = this->meanfield_df.get_weight()[n_spin - 1];
    auto &velocity = this->meanfield_df.get_velocity();
    auto &eigenvalues = this->meanfield_df.get_eigenvals();
    int nocc = 0;
    for (int i = 0; i != wg.size; i++)
    {
        if (wg.c[i] == 0.)
        {
            nocc = i;
            break;
        }
    }
    double omega_ev = this->omega[iomega];  // * HA2EV;
    std::complex<double> wing_term = 0.0;

    // std::complex<double> tmp = 0.0;
    for (int ispin = 0; ispin != n_spin; ispin++)
    {
        for (int ik = 0; ik != nk; ik++)
        {
            for (int iocc = 0; iocc != nbands; iocc++)
            {
                for (int iunocc = iocc; iunocc != nbands; iunocc++)
                {
                    double egap = (eigenvalues[ispin](ik, iunocc) -
                                   eigenvalues[ispin](ik, iocc));  // * HA2EV;
                    if (iocc < nocc && iunocc >= nocc)
                    {
                        /*tmp += conj(this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                    velocity[ispin][ik][alpha](iunocc, iocc)) /
                               (omega_ev * omega_ev + egap * egap);*/

                        wing_term += conj(/*this->Coul_vector.at(lambda).at(mu) **/
                                          this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                          velocity[ispin][ik][alpha](iunocc, iocc)) /
                                     (omega_ev * omega_ev + egap * egap);

                        /*if (iocc == 5 && iunocc == 40 && alpha == 0 && mu == 0 && iomega == 0)
                        {
                            std::cout << "mu, ik: " << mu << "," << ik << std::endl;
                            std::cout << "C: " << std::scientific << std::setprecision(8)
                                      << conj(this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]])
                                      << std::endl;
                            std::cout << "p: " << std::scientific << std::setprecision(8)
                                      << conj(velocity[ispin][ik][alpha](iunocc, iocc))
                                      << std::endl;
                            std::cout << "E_m, E_n: " << std::scientific << std::setprecision(8)
                                      << eigenvalues[ispin](ik, iunocc) << "," << std::scientific
                                      << std::setprecision(8) << eigenvalues[ispin](ik, iocc)
                                      << std::endl;
                            std::cout << "C*p: "
                                      << conj(
                                              this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                              velocity[ispin][ik][alpha](iunocc, iocc)) /
                                             (omega_ev * omega_ev + egap * egap)
                                      << std::endl;
                        }*/
                    }
                    else if (iunocc < nocc && iocc >= nocc)
                    {
                        // for metal
                        wing_term += 0.0 * /*conj(this->Coul_vector.at(lambda).at(mu)) **/
                                     this->Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                     velocity[ispin][ik][alpha](iunocc, iocc) /
                                     (omega_ev * omega_ev + egap * egap);
                    }
                }
            }
        }

        /*if (alpha == 0 && lambda == 0 && iomega == 0)
        {
            std::cout << "wing(x,l=0,w=0,mu): " << mu << ", " << wing_term << std::endl;
        }*/
        // wing_term += conj(this->Coul_vector.at(lambda).at(mu)) * tmp;
    }
    return wing_term;
};

void diele_func::init_Cs()
{
    using LIBRPA::atomic_basis_abf;
    using LIBRPA::atomic_basis_wfc;
    using RI::Tensor;
    int nk = this->kfrac_band.size();
    const int n_atom = Cs_data.data_libri.size();

    for (int ik; ik != nk; ik++)
    {
        for (int I = 0; I != n_atom; I++)
        {
            for (int J = 0; J != n_atom; J++)
            {
                int n_mu_I = atomic_basis_abf.get_atom_nb(I);
                int n_ao_I = atomic_basis_wfc.get_atom_nb(I);
                int n_ao_J = atomic_basis_wfc.get_atom_nb(J);

                Vector3_Order<double> k_frac = kfrac_band[ik];
                const std::array<double, 3> k_array = {k_frac.x, k_frac.y, k_frac.z};
                size_t total = n_ao_I * n_ao_J * n_mu_I;
                std::complex<double> *Cs_in = new std::complex<double>[total]();
                matrix_m<std::complex<double>> mat(n_ao_I * n_ao_J, n_mu_I, Cs_in, MAJOR::ROW,
                                                   MAJOR::COL);
                const std::initializer_list<std::size_t> shape{static_cast<std::size_t>(n_mu_I),
                                                               static_cast<std::size_t>(n_ao_I),
                                                               static_cast<std::size_t>(n_ao_J)};
                this->Ctri_ij.data_libri[I][{J, k_array}] =
                    Tensor<std::complex<double>>(shape, mat.dataobj.data);
                delete[] Cs_in;
            }
        }
    }

    int nbands = this->meanfield_df.get_n_bands();
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    this->Ctri_mn.resize(n_abf);
    for (int mu = 0; mu != n_abf; mu++)
    {
        this->Ctri_mn.at(mu).resize(nbands);
        for (int m = 0; m != nbands; m++)
        {
            this->Ctri_mn.at(mu).at(m).resize(nbands);
            for (int n = 0; n != nbands; n++)
            {
                for (int ik = 0; ik != nk; ik++)
                {
                    this->Ctri_mn.at(mu).at(m).at(n).insert(std::make_pair(kfrac_band[ik], 0.0));
                }
            }
        }
    }
    // std::cout << "* Success: Initialize Ctri_ij and Ctri_mn.\n";
};

void diele_func::FT_R2k()
{
    using LIBRPA::atomic_basis_abf;
    using LIBRPA::atomic_basis_wfc;
    int nk = this->kfrac_band.size();
    const int n_atom = Cs_data.data_libri.size();
    // std::cout << "Number of atom: " << n_atom << std::endl;

    for (int ik; ik != nk; ik++)
    {
        for (int I = 0; I != n_atom; I++)
        {
            for (int J = 0; J != n_atom; J++)
            {
                int n_mu_I = atomic_basis_abf.get_atom_nb(I);
                int n_ao_I = atomic_basis_wfc.get_atom_nb(I);
                int n_ao_J = atomic_basis_wfc.get_atom_nb(J);
#pragma omp parallel for schedule(dynamic) collapse(3)
                for (int mu = 0; mu != n_mu_I; mu++)
                {
                    for (int i = 0; i != n_ao_I; i++)
                    {
                        for (int j = 0; j != n_ao_J; j++)
                        {
                            Vector3_Order<double> k_frac = kfrac_band[ik];
                            const std::array<double, 3> k_array = {k_frac.x, k_frac.y, k_frac.z};
                            this->Ctri_ij.data_libri[I][{J, k_array}](mu, i, j) =
                                compute_Cijk(mu, I, i, J, j, ik);
                            /*if (ik == 19 && I == 1 && J == 0 && i == 0 && j == 1)
                            {
                                std::cout << "Cij: " << mu << ", "
                                          << this->Ctri_ij.data_libri[I][{J, k_array}](mu, i, j)
                                          << std::endl;
                            }*/
                        }
                    }
                }
            }
        }
    }
    /* Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);
    std::cout << "Number of Bvk cell: " << Rlist.size() << std::endl; */
    std::cout << "* Success: Fourier transform from Cs(R) to Cs(k).\n";
};

std::complex<double> diele_func::compute_Cijk(int mu, int I, int i, int J, int j, int ik)
{
    std::complex<double> Cijk = 0.0;
    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);
    Vector3_Order<double> k_frac = kfrac_band[ik];
    for (auto outer : Cs_data.data_libri[I])
    {
        auto J_Ra = outer.first;
        auto Ra = J_Ra.second;
        Vector3_Order<double> R = {Ra[0], Ra[1], Ra[2]};
        double ang = k_frac * R * TWO_PI;
        complex<double> kphase = complex<double>(cos(ang), sin(ang));
        if (J_Ra.first == J)
        {
            // std::cout << I << "," << J << "," << Ra[0] << "," << Ra[1] << "," << Ra[2] <<
            // std::endl;
            Cijk += kphase * Cs_data.data_libri[I][{J, Ra}](mu, i, j);
        }
    }
    return Cijk;
};

void diele_func::Cs_ij2mn()
{
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    int nk = this->kfrac_band.size();
    int nbands = this->meanfield_df.get_n_bands();

#pragma omp parallel for schedule(dynamic) collapse(4)
    for (int ik = 0; ik != nk; ik++)
    {
        for (int m = 0; m != nbands; m++)
        {
            for (int n = 0; n != nbands; n++)
            {
                for (int mu = 0; mu != n_abf; mu++)
                {
                    this->Ctri_mn.at(mu).at(m).at(n).at(kfrac_band[ik]) =
                        compute_Cs_ij2mn(mu, m, n, ik);
                    /*if (ik == 26 && m == 5 && n == 40)
                    {
                        lib_printf("Cmn: %5d, %15.5e, %15.5e\n", mu,
                                   Ctri_mn.at(mu).at(m).at(n).at(kfrac_band[ik]).real(),
                                   Ctri_mn.at(mu).at(m).at(n).at(kfrac_band[ik]).imag());
                    }*/
                }
            }
        }
    }

    std::cout << "* Success: transform of Cs^mu_ij(k) to Cs^mu_mn(k).\n";
};

std::complex<double> diele_func::compute_Cs_ij2mn(int mu, int m, int n, int ik)
{
    using LIBRPA::atomic_basis_abf;
    using LIBRPA::atomic_basis_wfc;
    const std::array<double, 3> k_array = {kfrac_band[ik].x, kfrac_band[ik].y, kfrac_band[ik].z};
    std::complex<double> total = 0.0;
    int Mu = atomic_basis_abf.get_i_atom(mu);
    int mu_local = atomic_basis_abf.get_local_index(mu, Mu);
    const int n_atom = Ctri_ij.data_libri.size();
    const int n_ao_Mu = atomic_basis_wfc.get_atom_nb(Mu);
    const ComplexMatrix eigenvectors = meanfield_df.get_eigenvectors()[0][ik];  // spin=1 only
    // #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i != n_ao_Mu; i++)
    {
        for (int J = 0; J != n_atom; J++)
        {
            int n_ao_J = atomic_basis_wfc.get_atom_nb(J);
            for (int j = 0; j != n_ao_J; j++)
            {
                std::complex<double> term1 =
                    conj(eigenvectors(m, atomic_basis_wfc.get_global_index(Mu, i))) *
                    Ctri_ij.data_libri[Mu][{J, k_array}](mu_local, i, j) *
                    eigenvectors(n, atomic_basis_wfc.get_global_index(J, j));
                std::complex<double> term2 =
                    eigenvectors(n, atomic_basis_wfc.get_global_index(Mu, i)) *
                    conj(Ctri_ij.data_libri[Mu][{J, k_array}](mu_local, i, j)) *
                    conj(eigenvectors(m, atomic_basis_wfc.get_global_index(J, j)));
                // #pragma omp critical
                total += term1 + term2;
            }
        }
    }

    return total;
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
        if (Vq.count(Mu) == 0 || Vq.at(Mu).count(Nu) == 0 || Vq.at(Mu).at(Nu).count(q) == 0)
            continue;
        const auto &Vq0 = Vq.at(Mu).at(Nu).at(q);
        const auto n_mu = LIBRPA::atomic_basis_abf.get_atom_nb(Mu);
        const auto n_nu = LIBRPA::atomic_basis_abf.get_atom_nb(Nu);
        std::valarray<complex<double>> Vq_va(Vq0->c, Vq0->size);
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
    for (int iv = n_abf - 2; iv != n_abf - n_nonsingular - 1; iv--)
    {
        // Here eigen solved by Scalapack is ascending order,
        // however, what we want is descending order.
        this->Coul_value.push_back(eigenvalues.c[iv]);  // throw away the largest one
        std::vector<std::complex<double>> newRow;

        // for (int jabf = n_abf - 1; jabf != -1; jabf--)
        for (int jabf = 0; jabf != n_abf; jabf++)
        {
            // newRow.push_back(coul_eigen_block.dataobj[jabf * n_abf + iv]);
            newRow.push_back(coul_eigen_block.dataobj[iv * n_abf + jabf]);
            //  if (coul_eigen_block.dataobj[iv * n_abf + jabf].imag() != 0.0)
            //      std::cout << "Imagine of Coulomb vector: " << iv << ", " << jabf << ", "
            //                << coul_eigen_block.dataobj[iv * n_abf + jabf] << std::endl;
            //   newRow.push_back(coul_eigen_block.dataobj[jabf * n_abf + iv]);
        }
        this->Coul_vector.push_back(newRow);
    }
    // std::reverse(this->Coul_value.begin(), this->Coul_value.end());
    // std::reverse(this->Coul_vector.begin(), this->Coul_vector.end());

    std::cout << "The largest/smallest eigenvalue of Coulomb matrix(non-singular): "
              << this->Coul_value.front() << ", " << this->Coul_value.back() << std::endl;
    std::cout << "The 1st/2nd/3rd/-1th eigenvalue of Coulomb matrix(Full): "
              << eigenvalues.c[n_abf - 1] << ", " << eigenvalues.c[n_abf - 2] << ", "
              << eigenvalues.c[n_abf - 3] << ", " << eigenvalues.c[0] << std::endl;
    std::cout << "Dim of eigenvectors: " << coul_eigen_block.dataobj.nr() << ", "
              << coul_eigen_block.dataobj.nc() << std::endl;
    std::cout << "Coulomb vector: lambda=0" << std::endl;
    for (int j = 0; j != n_abf; j++)
    {
        std::cout << j << "," << coul_eigen_block.dataobj[(n_abf - 1) * n_abf + j] << std::endl;
    }
    std::cout << "Coulomb vector: lambda=1" << std::endl;
    for (int j = 0; j != n_abf; j++)
    {
        std::cout << j << "," << coul_eigen_block.dataobj[(n_abf - 2) * n_abf + j] << std::endl;
    }
    std::cout << "* Success: diagonalize Coulomb matrix in the ABFs repre.\n";
};

void diele_func::test_wing()
{
    const int n_abf = LIBRPA::atomic_basis_abf.nb_total;
    std::cout << "BEGIN test wing !!!!!!!!!!" << std::endl;
    /*std::cout << "wing(z, mu=0) vs omega" << std::endl;
    for (int iomega = 0; iomega != this->omega.size(); iomega++)
    {
        std::complex<double> df = 0;
        // z direction and the first lambda
        df = this->wing_mu.at(2).at(0).at(iomega);

        std::cout << this->omega[iomega] << " " << df.real() << " " << df.imag() << std::endl;
    }*/
    std::cout << "wing_mu(z, iomega=0) vs mu" << std::endl;
    for (int mu = 0; mu != n_abf; mu++)
    {
        std::complex<double> df = 0;
        // z direction
        df = this->wing_mu.at(2).at(mu).at(0);

        std::cout << mu << " " << df.real() << " " << df.imag() << std::endl;
    }
    std::cout << "wing_lambda(z, iomega=0) vs lambda" << std::endl;
    for (int lambda = 0; lambda != n_nonsingular - 1; lambda++)
    {
        std::complex<double> df = 0;
        // z direction
        df = this->wing.at(2).at(lambda).at(0);

        std::cout << lambda << " " << df.real() << " " << df.imag() << std::endl;
    }
    std::cout << "END test wing !!!!!!!!!!" << std::endl;
    std::exit(0);
};