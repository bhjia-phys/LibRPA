// General SOC strict-2D optical GW against independent full dielectric inverses.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>
#include <vector>
#include "../api/dataset.h"
#include "../api/dataset_helper.h"
#include "../api/instance_manager.h"
#include "../core/chi0.h"
#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "librpa.hpp"
using namespace librpa_int;
using Complex = std::complex<double>;
namespace {
struct Spectrum
{
    std::vector<std::vector<double>> energies;
    std::vector<Matz> vectors;
};

Spectrum make_spectrum(const int nk, const double bz, const bool rotate_spin)
{
    Spectrum result;
    Matz spin_rotation(6, 6, MAJOR::COL);
    spin_rotation.zero_out();
    const double angle = rotate_spin ? 0.71 : 0.0;
    const auto off_diagonal = std::polar(std::sin(angle / 2), 0.37);
    for (int a = 0; a != 3; ++a)
    {
        spin_rotation(a, a) = spin_rotation(a + 3, a + 3) = std::cos(angle / 2);
        spin_rotation(a, a + 3) = -std::conj(off_diagonal);
        spin_rotation(a + 3, a) = off_diagonal;
    }
    for (int k = 0; k != nk; ++k)
    {
        Matz h(6, 6, MAJOR::COL);
        h.zero_out();
        const double phase = TWO_PI * k / nk;
        for (int a = 0; a != 3; ++a)
        {
            h(a, a) = bz;
            h(a + 3, a + 3) = -bz;
            h(a, a + 3) = std::polar(1.0, -TWO_PI * a / 3);
            h(a + 3, a) = std::conj(h(a, a + 3));
            for (int s = 0; s != 2; ++s)
            {
                const int next = (a + 1) % 3;
                // The periodic case has a nonconstant occupied projector and
                // an odd-k orbital term; its q/-q responses need not coincide.
                const Complex hop{0.15 + (nk > 1 ? 0.025 * std::cos(phase) : 0.0),
                                  s == 0 ? 0.04 : -0.04};
                h(a + 3 * s, next + 3 * s) = hop;
                h(next + 3 * s, a + 3 * s) = std::conj(hop);
                if (nk > 1) h(a + 3 * s, a + 3 * s) += 0.04 * (a - 1) * std::sin(phase);
            }
        }
        h = spin_rotation * h * transpose(spin_rotation, true);
        std::vector<double> eigenvalues;
        Matz eigenvectors;
        eigsh(h, eigenvalues, eigenvectors);
        assert(eigenvalues[3] - eigenvalues[2] > 1.0);
        result.energies.push_back(std::move(eigenvalues));
        result.vectors.push_back(std::move(eigenvectors));
    }
    return result;
}

std::vector<double> pack_operator(const Matz& value)
{
    std::vector<double> result(72);
    for (int i = 0; i != 3; ++i) for (int s = 0; s != 2; ++s)
        for (int j = 0; j != 3; ++j) for (int t = 0; t != 2; ++t)
        {
            const int index = 2 * ((2 * i + s) * 6 + 2 * j + t);
            const auto z = value(i + 3 * s, j + 3 * t);
            result[index] = z.real(); result[index + 1] = z.imag();
        }
    return result;
}

std::vector<Complex> gather(const Matz& tile, const ArrayDesc& desc, const MpiCommHandler& comm)
{
    std::vector<Complex> output(desc.m() * desc.n(), 0.0);
    for (int i = 0; i != tile.nr(); ++i) for (int j = 0; j != tile.nc(); ++j)
        output[desc.n() * desc.indx_l2g_r(i) + desc.indx_l2g_c(j)] = tile(i, j);
    MPI_Allreduce(MPI_IN_PLACE, output.data(), output.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm.comm);
    return output;
}


std::array<Complex, 9> inverse3(const std::array<Complex, 9>& a)
{
    const Complex determinant=a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);
    if(std::abs(determinant)<1e-12) throw std::runtime_error("singular independent strict-2D dielectric");
    return {(a[4]*a[8]-a[5]*a[7])/determinant,(a[2]*a[7]-a[1]*a[8])/determinant,(a[1]*a[5]-a[2]*a[4])/determinant,
            (a[5]*a[6]-a[3]*a[8])/determinant,(a[0]*a[8]-a[2]*a[6])/determinant,(a[2]*a[3]-a[0]*a[5])/determinant,
            (a[3]*a[7]-a[4]*a[6])/determinant,(a[1]*a[6]-a[0]*a[7])/determinant,(a[0]*a[4]-a[1]*a[3])/determinant};
}

std::vector<std::pair<double,double>> radial_rule(int count)
{
    std::vector<std::pair<double,double>> result;
    for(int j=0;j<count;++j)
    {
        double x=std::cos(PI*(j+.75)/(count+.5)), derivative=0;
        for(int iter=0;iter<40;++iter)
        {
            double p0=1,p1=x;
            for(int l=2;l<=count;++l){const double next=((2*l-1)*x*p1-(l-1)*p0)/l;p0=p1;p1=next;}
            derivative=count*(x*p1-p0)/(x*x-1);
            const double delta=p1/derivative;x-=delta;if(std::abs(delta)<1e-15)break;
        }
        result.emplace_back((x+1)/2,1/((1-x*x)*derivative*derivative));
    }
    return result;
}

Matz independent_average(const std::vector<double>& energies,const Matz& c,
                         const std::array<Matz,3>& r,const Matz& u,double omega,int radial_count,
                         bool impose_wrong_left=false)
{
    constexpr double area=7.1*5.9;
    const double alpha=TWO_PI/area;
    Matz p(3,3,MAJOR::COL), h(3,3,MAJOR::COL), zr(3,3,MAJOR::COL), zl(3,3,MAJOR::COL);
    for(int n=0;n<6;++n)for(int m=0;m<6;++m)
    {
        const int df=int(n<3)-int(m<3);if(!df)continue;
        const Complex factor=double(df)/Complex{energies[n]-energies[m],omega};
        std::array<Complex,3> d{}, hv{};
        for(int mu=0;mu<3;++mu)for(int s=0;s<2;++s)
            d[mu]+=std::conj(c(mu+3*s,n))*c(mu+3*s,m);
        for(int a=0;a<3;++a)for(int i=0;i<6;++i)
            hv[a]+=Complex{0,-1}*std::conj(c(i,n))*r[a](i,i)*c(i,m);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        {
            p(i,j)+=factor*d[i]*std::conj(d[j]);
            h(i,j)-=alpha*factor*hv[i]*std::conj(hv[j]);
            zr(i,j)-=std::sqrt(alpha)*factor*d[i]*std::conj(hv[j]);
            zl(i,j)-=std::sqrt(alpha)*factor*hv[i]*std::conj(d[j]);
        }
    }
    if(impose_wrong_left) zl=transpose(zr,true);
    Matz body(2,2,MAJOR::COL), right(2,3,MAJOR::COL), left(3,2,MAJOR::COL);
    const std::array<double,2> roots{std::sqrt(1.1),std::sqrt(.7)};
    for(int a=0;a<2;++a)for(int b=0;b<2;++b)
    {
        body(a,b)=a==b?1.:0.;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            body(a,b)-=roots[a]*std::conj(u(i,a+1))*p(i,j)*u(j,b+1)*roots[b];
    }
    for(int a=0;a<2;++a)for(int axis=0;axis<3;++axis)for(int mu=0;mu<3;++mu)
    {
        right(a,axis)+=roots[a]*std::conj(u(mu,a+1))*zr(mu,axis);
        left(axis,a)+=zl(axis,mu)*u(mu,a+1)*roots[a];
    }
    const auto radial=radial_rule(radial_count);
    Matz average(3,3,MAJOR::COL);
    // Direct reciprocal geometry of the actual oblique in-plane cell.
    // This independently bounds its Voronoi polygon; no native q_gamma helper.
    constexpr int angles=5000;
    const double gamma_area=TWO_PI*TWO_PI/area, angular_weight=TWO_PI/angles;
    for(int k=0;k<angles;++k)
    {
        const double nx=std::cos(TWO_PI*k/angles),ny=std::sin(TWO_PI*k/angles);
        double qmax=1e10;
        for(int a=-3;a<=3;++a)for(int b=-3;b<=3;++b)if(a||b)
        {
            const double gx=TWO_PI*a/7.1,gy=TWO_PI*(-1.3*a/(7.1*5.9)+b/5.9);
            const double dot=nx*gx+ny*gy;
            if(dot>1e-13)qmax=std::min(qmax,(gx*gx+gy*gy)/(2*dot));
        }
        const Complex hh=nx*(h(0,0)*nx+h(0,1)*ny)+ny*(h(1,0)*nx+h(1,1)*ny);
        for(const auto& [fraction,weight]:radial)
        {
            const double q=fraction*qmax, sq=std::sqrt(q);
            std::array<Complex,9> epsilon{};epsilon[0]=1.+q*hh;
            for(int a=0;a<2;++a)
            {
                epsilon[3*(a+1)]=sq*(right(a,0)*nx+right(a,1)*ny);
                epsilon[a+1]=sq*(nx*left(0,a)+ny*left(1,a));
                for(int b=0;b<2;++b)epsilon[3*(a+1)+b+1]=body(a,b);
            }
            const auto inverse=inverse3(epsilon);
            // Unit-integral auxiliary probes have total charge norm squared3.
            // This external leg follows that charge, not the native metadata helper.
            const std::array<double,3> legs{std::sqrt(3/area)*std::sqrt(TWO_PI/q),roots[0],roots[1]};
            const double measure=angular_weight*weight*fraction*qmax*qmax/gamma_area;
            for(int a=0;a<3;++a)for(int b=0;b<3;++b)
                average(a,b)+=measure*legs[a]*(inverse[3*a+b]-(a==b?1.:0.))*legs[b];
        }
    }
    return u*average*transpose(u,true);
}
void check_native()
{
    constexpr double bz=.21, angle=0, height=18, basis_scale=1;
    const std::string directory=".";
    librpa::Handler handler(MPI_COMM_WORLD);
    auto ds_ptr = api::get_dataset_instance(handler);
    auto& ds = *ds_ptr;
    const auto spectrum = make_spectrum(1, bz, false);
    const auto& e = spectrum.energies.front();
    const auto& c = spectrum.vectors.front();
    ds.basis_wfc.set({1, 1, 1}); ds.basis_aux.set({1, 1, 1});
    ds.mf.set(1, 1, 6, 3, 2);
    ds.mf.get_efermi() = (e[2] + e[3]) * 0.5;
    const double ca = std::cos(angle), sa = std::sin(angle);
    const std::vector<double> lattice{7.1 * ca, 7.1 * sa, 0,
                                      1.3 * ca - 5.9 * sa, 1.3 * sa + 5.9 * ca, 0,
                                      0, 0, height};
    ds.pbc.set_latvec(lattice);
    ds.pbc.set_kgrids_kvec(1, 1, 1, {0, 0, 0});
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 1);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds.desc_abf = ArrayDesc(ds.blacs_h); ds.desc_abf.init_square_blk(3, 3, 0, 0);
    handler.set_basis_convention(-1, 0, LIBRPA_ANGULAR_ORDER_ABS_PM,
                                 LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M);
    for (int s = 0; s != 2; ++s)
    {
        ComplexMatrix block(6, 3);
        for (int n = 0; n != 6; ++n) for (int a = 0; a != 3; ++a) block(n, a) = c(a + 3 * s, n);
        ds.mf.get_eigenvectors()[0][s][0] = block;
    }
    for (int n = 0; n != 6; ++n)
    {
        ds.mf.get_eigenvals()[0](0, n) = e[n];
        ds.mf.get_weight()[0](0, n) = n < 3 ? 1.0 : 0.0;
    }
    ds.cs_data.use_libri = true;
    if (ds.comm_h.is_root()) for (int a = 0; a != 3; ++a)
        ds.cs_data.data_libri[a][{a, {0, 0, 0}}] =
            RI::Tensor<double>({1, 1, 1}, std::make_shared<std::valarray<double>>(0.5 / basis_scale, 1));
    const auto assigned = dispatch_upper_triangular_tasks(3, ds.blacs_h.myid, ds.blacs_h.nprows,
        ds.blacs_h.npcols, ds.blacs_h.myprow, ds.blacs_h.mypcol);
    const std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    const auto q = ds.pbc.klist.front();
    Matz u(3, 3, MAJOR::COL), v(3, 3, MAJOR::COL), sqrtveig(3, 3, MAJOR::COL);
    const std::array<double, 3> eigenvalues{20.0 * basis_scale * basis_scale, 1.1 * basis_scale * basis_scale, 0.7 * basis_scale * basis_scale};
    for (int i = 0; i != 3; ++i) u(i, 0) = 1 / std::sqrt(3.0);
    u(0, 1) = 1 / std::sqrt(2.0); u(1, 1) = -1 / std::sqrt(2.0);
    u(0, 2) = u(1, 2) = 1 / std::sqrt(6.0); u(2, 2) = -2 / std::sqrt(6.0);
    for (int i = 0; i != 3; ++i) for (int j = 0; j != 3; ++j)
    {
        sqrtveig(i, j) = u(i, j) * std::sqrt(eigenvalues[j]);
        for (int a = 0; a != 3; ++a) v(i, j) += u(i, a) * eigenvalues[a] * std::conj(u(j, a));
    }
    for (const auto& [a, b] : pairs)
    {
        auto block = std::make_shared<ComplexMatrix>(1, 1); (*block)(0, 0) = v(a, b);
        ds.vq[a][b][q] = block;
    }
    Matz h(6, 6, MAJOR::COL), overlap(6, 6, MAJOR::COL);
    std::array<Matz, 3> r{Matz(6, 6, MAJOR::COL), Matz(6, 6, MAJOR::COL), Matz(6, 6, MAJOR::COL)};
    for (int i = 0; i != 6; ++i)
    {
        overlap(i, i) = 1;
        for (int j = 0; j != 6; ++j) for (int n = 0; n != 6; ++n)
            h(i, j) += c(i, n) * e[n] * std::conj(c(j, n));
        const double theta = TWO_PI * (i % 3) / 3 + angle;
        r[0](i, i) = 0.8 * std::cos(theta); r[1](i, i) = 0.8 * std::sin(theta);
    }
    auto hp = pack_operator(h), sp = pack_operator(overlap);
    std::array<std::vector<double>, 3> rp{pack_operator(r[0]), pack_operator(r[1]), pack_operator(r[2])};
    LibrpaLocalizedBlock hb{{0, 0, 0}, hp.data(), 1}, sb{{0, 0, 0}, sp.data(), 1};
    std::array<LibrpaLocalizedBlock, 3> rb;
    LibrpaLocalizedOperator hop{6, 1, &hb};
    LibrpaNativeVelocityInput input{};
    input.n_spins = 1; input.hamiltonian = &hop; input.overlap = {6, 1, &sb};
    for (int a = 0; a != 3; ++a)
    {
        rb[a] = {{0, 0, 0}, rp[a].data(), 1}; input.position[a] = {6, 1, &rb[a]};
    }
    input.inplane_area_bohr2 = 7.1 * 5.9;
    // ABACUS l=0 multipole is the radial moment, not integrated charge:
    // Q = sqrt(4*pi)*moment. Each unscaled auxiliary probe has Q=1.
    const double radial_monopole = basis_scale / std::sqrt(4 * PI);
    input.auxiliary_monopole_norm_squared = 3 * radial_monopole * radial_monopole;
    librpa::Options options;
    options.nfreq = 24; options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    options.minimax_emin = .4; options.minimax_emax = 4; options.minimax_regulation = 1e-12;
    options.parallel_routing = LIBRPA_ROUTING_LIBRI;
    options.use_2d_dielectric = options.replace_w_head = options.use_scalapack_gw_wc = LIBRPA_SWITCH_ON;
    options.option_dielect_func = 3; options.sqrt_coulomb_threshold = 0;
    options.set_output_dir(directory.c_str());
    handler.prepare_native_velocity(options, ds.comm_h.is_root() ? &input : nullptr, true);
    Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
             ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
    chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    chi.build(LIBRPA_ROUTING_LIBRI, ds.cs_data, pairs, ds.basis_aux, no_shrink, ds.blacs_h);
    const int nf = ds.tfg.get_n_grids();
    std::vector<Complex> chi_full(nf * 9, 0), inner, caller;
    for (const auto& [omega, qs] : chi.get_chi0_q()) for (const auto& [momentum, rows] : qs)
        for (const auto& [i, cols] : rows) for (const auto& [j, block] : cols)
            chi_full[9 * ds.tfg.get_freq_index(omega) + 3 * i + j] = block(0, 0);
    MPI_Allreduce(MPI_IN_PLACE, chi_full.data(), chi_full.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, ds.comm_h.comm);

    const std::vector<Complex> epsmac(ds.epsmacs_imagfreq.begin(),ds.epsmacs_imagfreq.end());
    if(epsmac.size()!=std::size_t(nf))throw std::runtime_error("missing native optical activation");
    auto wc_coulomb=ds.vq;
    const auto wc=compute_Wc_freq_q_blacs(chi,ds.vq,wc_coulomb,0,true,3,epsmac,
        ds.p_headwing.get(),ds.blacs_h,ds.desc_abf,false,".",false,false,false);
    double error=0,radial_error=0,wrong_left=0,odd_response=0;
    for(int f=0;f<nf;++f)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        odd_response=std::max(odd_response,std::abs(chi_full[9*f+3*i+j]-std::conj(chi_full[9*f+3*j+i])));
    for(int f:{0,16})
    {
        const auto frequency=ds.tfg.get_freq_nodes()[f];
        const auto native=gather(wc.at(frequency).at(q),ds.desc_abf,ds.comm_h);
        const auto exact=independent_average(e,c,r,u,frequency,32);
        const auto coarse=independent_average(e,c,r,u,frequency,16);
        const auto wrong=independent_average(e,c,r,u,frequency,16,true);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        {
            error=std::max(error,std::abs(native[3*i+j]-exact(i,j)));
            radial_error=std::max(radial_error,std::abs(coarse(i,j)-exact(i,j)));
            wrong_left=std::max(wrong_left,std::abs(wrong(i,j)-exact(i,j)));
        }
    }
    if(ds.comm_h.is_root())std::cout<<"Strict2D general native: W="<<error<<" radial16/32="<<radial_error
        <<" incorrect-left="<<wrong_left<<" odd-response="<<odd_response<<std::endl;
    if(error>1e-10||radial_error>2e-12||wrong_left<1e-9||odd_response<1e-6)
        throw std::runtime_error("general strict-2D native screening oracle failed");
}
} // namespace
int main(int argc,char** argv)
{
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    global::init_global_mpi(MPI_COMM_WORLD);global::init_global_io();
    try{check_native();}catch(const std::exception& ex){std::cerr<<ex.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
    global::finalize_global_io();global::finalize_global_mpi();MPI_Finalize();
}
