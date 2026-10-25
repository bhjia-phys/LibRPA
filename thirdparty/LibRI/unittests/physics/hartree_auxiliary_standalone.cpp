#include "RI/physics/Hartree.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

// One atom, one real AO/auxiliary function, two k points. The closed-form
// charge is M=2[(0.3+0.2)*0.6+(0.3-0.2)*0.8]=0.76.
// Native normalization gives N=2.7*M/2=1.026, H(k)=2*C(k)*N.
// Check both the exposed potential and unchanged ordinary-cal_hartree result.
int main(int argc, char** argv)
{
    MPI_Init(&argc,&argv);
    int rank = 0, size = 0, failed = 0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    try
    {
        if (size > 2) throw std::runtime_error("Use one or two ranks for this control");
        using Z = std::complex<double>;
        using T = RI::Tensor<Z>;
        using H = RI::Hartree<int,int,3,Z>;
        using C = std::array<int,3>;
        using Map = std::map<int,std::map<std::pair<int,C>,T>>;
        const auto prepare = [&](H& h)
        {
            h.init({{{0.0,0.0,0.0}},{{0.5,0.0,0.0}}});
            h.lrik.mpi_comm = MPI_COMM_WORLD; h.lrik.period = {{2,1,1}};
            h.list_I = {0}; h.list_J = {0}; h.list_IJ = {0};
            for (int k = rank; k < 2; k += size) h.k_indices.push_back(k);
            Map cs,vs;
            cs[0][{0,{{0,0,0}}}] = T({1,1,1}); cs[0][{0,{{0,0,0}}}](0,0,0) = 0.3;
            cs[0][{0,{{1,0,0}}}] = T({1,1,1}); cs[0][{0,{{1,0,0}}}](0,0,0) = 0.2;
            vs[0][{0,{{0,0,0}}}] = T({1,1}); vs[0][{0,{{0,0,0}}}](0,0) = 2.7;
            const std::map<std::string,double> flags{{"flag_comm",false},{"flag_filter",false}};
            h.lrik.set_tensors_map2(cs,{RI::Label::ab::a,RI::Label::ab::b},flags,"Cs_");
            h.lrik.set_tensors_map2(vs,{RI::Label::ab::a0b0},flags,"Vs_");
        };
        std::map<int,std::map<int,std::map<int,T>>> density;
        for (int k = 0; k != 2; ++k)
        { density[0][0][k] = T({1,1}); density[0][0][k](0,0) = k == 0 ? 0.6 : 0.8; }
        H original; prepare(original); const auto baseline = original.cal_hartree(density);
        H exposed; prepare(exposed); std::map<int,T> potential;
        const auto actual = exposed.cal_hartree_with_auxiliary_potential(density,potential);
        double error = std::abs(potential.at(0).ptr()[0]-Z(1.026));
        for (int k : exposed.k_indices)
        {
            const Z expected = (k == 0 ? 1.0 : 0.2)*1.026;
            error = std::max(error,std::abs(actual.at(0).at(0).at(k)(0,0)-expected));
            error = std::max(error,std::abs(actual.at(0).at(0).at(k)(0,0)-
                                          baseline.at(0).at(0).at(k)(0,0)));
        }
        if (error > 2e-14) throw std::runtime_error("Auxiliary Hartree closed-form control failed");
        if (rank == 0) std::cout << "Hartree auxiliary control: ranks=" << size
                                 << " max_error=" << error << '\n';
    }
    catch (const std::exception& e)
    { std::cerr << e.what() << '\n'; failed = 1; }
    int any = 0; MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    MPI_Finalize(); return any;
}
