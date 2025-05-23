#include "read_data.h"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <unordered_map>

#include "atoms.h"
#include "atomic_basis.h"
#include "constants.h"
#include "matrix.h"
#include "ri.h"
#include "pbc.h"
#include "geometry.h"
#include "envs_mpi.h"
#include "envs_io.h"
#include "utils_io.h"
#include "stl_io_helper.h"
#include "profiler.h"

#include "librpa.h"
#include "utils_mem.h"

using std::ifstream;
using std::string;

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf)
{
    // cout << "Begin to read aims-band_out" << endl;
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    int n_kpoints, n_spins, n_bands, n_aos;
    double efermi;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_bands;
    infile >> n_aos;
    infile >> efermi;

    // TODO: replace it with set_dimension
    mf.set(n_spins, n_kpoints, n_bands, n_aos);

    // Load the file data
    auto eskb = new double [n_spins * n_kpoints * n_bands];
    auto wskb = new double [n_spins * n_kpoints * n_bands];

    const int n_kb = n_kpoints * n_bands;

    int iline = 6;

    //cout<<"|eskb: "<<endl;
    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " + to_string(iline) +
                                       ", file: " + file_path);
            }
            iline++;
            //cout<<ik<<is<<endl;
            int k_index = stoi(ks) - 1;
            // int s_index = stoi(ss) - 1;
            for (int i = 0; i != n_bands; i++)
            {
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " + to_string(iline) +
                                           ", file: " + file_path);
                }
            iline++;
                wskb[is * n_kb + k_index * n_bands + i] = stod(ws); // different with abacus!
                eskb[is * n_kb + k_index * n_bands + i] = stod(es);
                //cout<<" i_band: "<<i<<"    eskb: "<<eskb[is](k_index, i)<<endl;
            }
        }
    }
    // for (int is = 0; is != n_spins; is++)
    //     print_matrix("eskb_mat",eskb[is]);

    set_wg_ekb_efermi(n_spins, n_kpoints, n_bands, wskb, eskb, efermi);

    // free buffer
    delete [] eskb;
    delete [] wskb;
}

int read_vxc(const string &file_path, std::vector<matrix> &vxc)
{
    ifstream infile;
    infile.open(file_path);
    double ha, ev;
    int n_spins, n_kpoints, n_states;
    int retcode;

    // dimension information
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    if (!infile.good())
    {
        return 1;
    }

    vxc.clear();
    vxc.resize(n_spins);
    for (int is = 0; is != n_spins; is++)
    {
        vxc[is].create(n_kpoints, n_states);
    }

    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            for (int i = 0; i != n_states; i++)
            {
                infile >> ha >> ev;
                if (!infile.good())
                {
                    return 2;
                }
                vxc[is](ik, i) = ha;
            }
        }
    }
    return 0;
}

static int handle_KS_file(const string &file_path, MeanField &mf)
{
    int ret = 0;
    // cout<<file_path<<endl;
    ifstream infile;
    // cout << "Reading eigenvector from file " << file_path << endl;
    infile.open(file_path);
    if (!infile.good())
        return 1;

    string rvalue, ivalue, kstr;

    const auto nspin = mf.get_n_spins();
    const auto nband = mf.get_n_bands();
    const auto nao = mf.get_n_aos();
    const auto n = nband * nao;

    std::vector<double> re(nspin * nband * nao);
    std::vector<double> im(nspin * nband * nao);

    while (infile.peek() != EOF)
    {
        infile >> kstr;
        int ik = stoi(kstr) - 1;
        // cout<<"     ik: "<<ik<<endl;
        if (infile.peek() == EOF)
            break;
        // for aims !!!
        for (int iw = 0; iw != nao; iw++)
        {
            for (int ib = 0; ib != nband; ib++)
            {
                for (int is = 0; is != nspin; is++)
                {
                    // cout<<iw<<ib<<is<<ik;
                    infile >> rvalue >> ivalue;
                    if (infile.bad())
                    {
                        ret = 1;
                        break;
                    }
                    // cout<<rvalue<<ivalue<<endl;
                    re[is * n + ib * nao + iw] = stod(rvalue);
                    im[is * n + ib * nao + iw] = stod(ivalue);
                }
            }
        }
        for (int is = 0; is != nspin; is++)
        {
            set_ao_basis_wfc(is, ik, re.data() + is * n, im.data() + is * n);
        }
        // for abacus
        // for (int ib = 0; ib != NBANDS; ib++)
        //     for (int iw = 0; iw != NLOCAL; iw++)
        //         for (int is = 0; is != NSPIN; is++)
        //         {
        //             // cout<<iw<<ib<<is<<ik;
        //             infile >> rvalue >> ivalue;
        //             // cout<<rvalue<<ivalue<<endl;
        //             wfc_k.at(stoi(ik) - 1)(ib, iw) = complex<double>(stod(rvalue), stod(ivalue));
        //         }
    }
    return ret;
}

int read_eigenvector(const string &dir_path, MeanField &mf)
{
    // return code
    int ret = 0;
    int files_read = 0;

    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    vector<string> files;
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        // cout << fm << " find:" << fm.find("KS_eigenvector") << "\n";
        if (fm.find("KS_eigenvector") == 0)
        {
            ret = handle_KS_file(dir_path + fm, mf);
            if (ret != 0)
            {
                break;
            }
            files_read++;
        }
    }
    closedir(dir);
    dir = NULL;

    if (files_read == 0)
    {
        ret = -1;
    }

    //auto tmp_wfc=mf.get_eigenvectors();
    // for(int is=0;is!=mf.get_n_spins();is++)
    //     print_complex_matrix("wfc ",tmp_wfc.at(is).at(0));
    // cout << "Finish read KS_eignvector! " << endl;
    return ret;
}

//! Check if Cs data file is in ASCII text or unformatted binary format
static bool check_Cs_file_binary(const string &file_path)
{
    // Current strategy:
    //   Assume the file is ASCII, try to read to the first integer, which is the number of atoms
    //   If it succeeds, then the file is ASCII, otherwise it is unformatted.
    //
    // This is the simplest way, and gives less false positives than assuming the file is binary
    // and checking the first integer by reading the first 4 bytes with infile.read.
    bool is_binary = true;
    ifstream infile;
    int natom;
    // infile.open(file_path, std::ios::in | std::ios::binary);
    infile.open(file_path, std::ios::in);
    // infile.read((char *) &natom, sizeof(int));
    infile >> natom;
    if (infile.good())
    {
        is_binary = false;
    }
    // cout << natom << " " << is_binary << endl;
    infile.close();
    return is_binary;
}

//! Check if Coulomb matrix data file is in ASCII text or unformatted binary format
static bool check_coulomb_file_binary(const string &file_path)
{
    bool is_binary = true;
    ifstream infile;
    int nirk;
    infile.open(file_path, std::ios::in);
    infile >> nirk;
    if (infile.good())
    {
        is_binary = false;
    }
    // cout << nirk << " " << is_binary << endl;
    infile.close();
    return is_binary;
}

static size_t handle_Cs_file(const string &file_path, double threshold, const vector<atpair_t> &local_atpair)
{
    
    set<size_t> loc_atp_index;
    for(auto &lap:local_atpair)
    {
        loc_atp_index.insert(lap.first);
        loc_atp_index.insert(lap.second);
    }
    // cout<<"READING Cs from file: "<<file_path<<"  Cs_first_size: "<<loc_atp_index.size()<<endl;
    // map<size_t,map<size_t,map<Vector3_Order<int>,std::shared_ptr<matrix>>>> Cs_m;
    size_t cs_discard = 0;
    string natom_s, ncell_s, ia1_s, ia2_s, ic_1, ic_2, ic_3, i_s, j_s, mu_s, Cs_ele;
    int R[3];
    ifstream infile;
    infile.open(file_path);
    infile >> natom_s >> ncell_s;
    natom = stoi(natom_s);
    ncell = stoi(ncell_s);

    /* cout<<"  Natom  Ncell  "<<natom<<"  "<<ncell<<endl; */
    // for(int loop=0;loop!=natom*natom*ncell;loop++)
    while (infile.peek() != EOF)
    {
        infile >> ia1_s >> ia2_s >> ic_1 >> ic_2 >> ic_3 >> i_s;
        if (infile.peek() == EOF)
            break;
        // cout << " ia1_s,ia2_s: " << ia1_s << "  " << ia2_s << endl;
        infile >> j_s >> mu_s;
        // cout<<ic_1<<mu_s<<endl;
        int ia1 = stoi(ia1_s) - 1;
        int ia2 = stoi(ia2_s) - 1;
        R[0] = stoi(ic_1);
        R[1] = stoi(ic_2);
        R[2] = stoi(ic_3);
        int n_i = stoi(i_s);
        int n_j = stoi(j_s);
        int n_mu = stoi(mu_s);

        // cout<< ia1<<ia2<<box<<endl;
        shared_ptr<matrix> cs_ptr = make_shared<matrix>();
        cs_ptr->create(n_i * n_j, n_mu);
        // cout<<cs_ptr->nr<<cs_ptr->nc<<endl;

        for (int i = 0; i != n_i; i++)
            for (int j = 0; j != n_j; j++)
                for (int mu = 0; mu != n_mu; mu++)
                {
                    infile >> Cs_ele;
                    (*cs_ptr)(i * n_j + j, mu) = stod(Cs_ele);
                    // if (i == j)
                    // {
                    //     (*cs_ptr)(i * n_j + j, mu) = 1.0;
                    // }
                }
        // if(!loc_atp_index.count(ia1))
        //     continue;
        // if (box == Vector3_Order<int>({0, 0, 1}))continue;
        bool keep = loc_atp_index.count(ia1) && (*cs_ptr).absmax() >= threshold;
        set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, cs_ptr->c, int(!keep));
        // cout<<cs_ptr->nr<<cs_ptr->nc<<endl;
        if (!keep)
        {
            cs_discard++;
        }
    }
    infile.close();
    return cs_discard;
}

static size_t handle_Cs_file_binary(const string &file_path, double threshold, const vector<atpair_t> &local_atpair)
{
    
    set<size_t> loc_atp_index;
    for(auto &lap:local_atpair)
    {
        loc_atp_index.insert(lap.first);
        loc_atp_index.insert(lap.second);
    }
    // cout<<"READING Cs from file: "<<file_path<<"  Cs_first_size: "<<loc_atp_index.size()<<endl;
    // map<size_t,map<size_t,map<Vector3_Order<int>,std::shared_ptr<matrix>>>> Cs_m;
    size_t cs_discard = 0;
    ifstream infile;
    int dims[8];
    int n_apcell_file;

    infile.open(file_path, std::ios::in | std::ios::binary);
    infile.read((char *) &natom, sizeof(int));
    infile.read((char *) &ncell, sizeof(int));
    infile.read((char *) &n_apcell_file, sizeof(int));

    int R[3];

    for (int i = 0; i < n_apcell_file; i++)
    {
        infile.read((char *) &dims[0], 8 * sizeof(int));
        // cout<<ic_1<<mu_s<<endl;
        int ia1 = dims[0] - 1;
        int ia2 = dims[1] - 1;
        R[0] = dims[2];
        R[1] = dims[3];
        R[2] = dims[4];
        int n_i = dims[5];
        int n_j = dims[6];
        int n_mu = dims[7];

        // cout<< ia1<<ia2<<box<<endl;

        shared_ptr<matrix> cs_ptr = make_shared<matrix>();
        cs_ptr->create(n_i * n_j, n_mu);
        infile.read((char *) cs_ptr->c, n_i * n_j * n_mu * sizeof(double));
        bool keep = loc_atp_index.count(ia1) && (*cs_ptr).absmax() >= threshold;
        // cout << (*cs_ptr).absmax() << "\n";
        set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, cs_ptr->c, int(!keep));
        // cout<<cs_ptr->nr<<cs_ptr->nc<<endl;
        if (!keep)
        {
            cs_discard++;
        }
    }
    return cs_discard;
}

size_t read_Cs(const string &dir_path, double threshold,const vector<atpair_t> &local_atpair)
{
    size_t cs_discard = 0;
    // cout << "Begin to read Cs" << endl;
    // cout << "cs_threshold:  " << threshold << endl;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    vector<string> files;
    bool binary;
    bool binary_checked = false;

    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (fm.find("Cs_data") == 0)
        {
            const auto fn = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_Cs_file_binary(fn);
                binary_checked = true;
                if (LIBRPA::envs::myid_global == 0)
                {
                    if (binary)
                    {
                        cout << "Unformatted binary Cs files detected" << endl;
                    }
                    else
                    {
                        cout << "ASCII format Cs files detected" << endl;
                    }
                }
            }
            if (binary)
            {
                cs_discard += handle_Cs_file_binary(fn, threshold, local_atpair);
            }
            else
            {
                cs_discard += handle_Cs_file(fn, threshold, local_atpair);
            }
        }
    }
    closedir(dir);
    dir = NULL;
    // initialize basis set object
    LIBRPA::atomic_basis_wfc.set(atom_nw);
    LIBRPA::atomic_basis_abf.set(atom_mu);
    
    // atom_mu_part_range.resize(atom_mu.size());
    // atom_mu_part_range[0]=0;
    // for(int I=1;I!=atom_mu.size();I++)
    //     atom_mu_part_range[I]=atom_mu.at(I-1)+atom_mu_part_range[I-1];
    
    // N_all_mu=atom_mu_part_range[natom-1]+atom_mu[natom-1];
    init_N_all_mu();

    // for(int i=0;i!=atom_mu_part_range.size();i++)
    //     cout<<" atom_mu_part_range ,i: "<<i<<"    "<<atom_mu_part_range[i]<<endl;

    // cout << "Finish read Cs" << endl;
    return cs_discard;
}

std::vector<size_t> handle_Cs_file_dry(const string &file_path, double threshold)
{
    std::vector<size_t> Cs_ids_keep;
    string natom_s, ncell_s, ia1_s, ia2_s, ic_1, ic_2, ic_3, i_s, j_s, mu_s, Cs_ele;
    ifstream infile;
    infile.open(file_path);
    infile >> natom_s >> ncell_s;
    natom = stoi(natom_s);
    ncell = stoi(ncell_s);

    size_t id = 0;
    while (infile.peek() != EOF)
    {
        infile >> ia1_s;
        if (infile.peek() == EOF)
            break;
        infile >> ia2_s >> ic_1 >> ic_2 >> ic_3 >> i_s >> j_s >> mu_s;
        int n_i = stoi(i_s);
        int n_j = stoi(j_s);
        int n_mu = stoi(mu_s);

        double maxval = -1.0;
        for (int i = 0; i != n_i; i++)
            for (int j = 0; j != n_j; j++)
                for (int mu = 0; mu != n_mu; mu++)
                {
                    infile >> Cs_ele;
                    maxval = std::max(maxval, abs(stod(Cs_ele)));
                }
        LIBRPA::envs::ofs_myid << id << " (" << ic_1 << "," << ic_2 << "," << ic_3 << ") " << maxval << " keep? " << (maxval >= threshold) << endl;
        if (maxval >= threshold)
            Cs_ids_keep.push_back(id);
        id++;
    }
    LIBRPA::envs::ofs_myid << file_path << ": " << Cs_ids_keep << endl;
    infile.close();
    return Cs_ids_keep;
}

std::vector<size_t> handle_Cs_file_binary_dry(const string &file_path, double threshold)
{
    std::vector<size_t> Cs_ids_keep;
    ifstream infile;
    int dims[8];
    int n_apcell_file;
    int n_processed = 0;

    infile.open(file_path, std::ios::in | std::ios::binary);
    infile.read((char *) &natom, sizeof(int));
    infile.read((char *) &ncell, sizeof(int));
    infile.read((char *) &n_apcell_file, sizeof(int));

    for (int i_file = 0; i_file < n_apcell_file; i_file++)
    {
        infile.read((char *) &dims[0], 8 * sizeof(int));
        // cout<<ic_1<<mu_s<<endl;
        // const int ia1 = dims[0] - 1;
        // const int ia2 = dims[1] - 1;
        // const int ic1 = dims[2];
        // const int ic2 = dims[3];
        // const int ic3 = dims[4];
        const int n_i = dims[5];
        const int n_j = dims[6];
        const int n_mu = dims[7];

        matrix mat(n_i * n_j, n_mu);
        infile.read((char *) mat.c, n_i * n_j * n_mu * sizeof(double));
        double maxval = mat.absmax();
        n_processed++;
        if (maxval >= threshold)
        {
            Cs_ids_keep.push_back(i_file);
#ifdef LIBRPA_DEBUG
            // LIBRPA::envs::ofs_myid << i_file << " (" << ic1 << "," << ic2 << "," << ic3 << ") " << maxval << " kept, maxval: " << maxval << endl;
#endif
        }
    }
    // LIBRPA::envs::ofs_myid << file_path << ": kept " << Cs_ids_keep.size() << " of " << n_processed << endl;
#ifdef LIBRPA_DEBUG
    // LIBRPA::envs::ofs_myid << Cs_ids_keep << endl;
#endif
    infile.close();
    return Cs_ids_keep;
}

static size_t handle_Cs_file_by_ids(const string &file_path, double threshold, const vector<size_t> &ids)
{
    size_t cs_discard = 0;
    string natom_s, ncell_s, ia1_s, ia2_s, ic_1, ic_2, ic_3, i_s, j_s, mu_s, Cs_ele;
    ifstream infile;
    infile.open(file_path);
    infile >> natom_s >> ncell_s;
    natom = stoi(natom_s);
    ncell = stoi(ncell_s);
    /* cout<<"  Natom  Ncell  "<<natom<<"  "<<ncell<<endl; */
    // for(int loop=0;loop!=natom*natom*ncell;loop++)
    size_t id = 0;
    int R[3];

    while (infile.peek() != EOF)
    {
        infile >> ia1_s >> ia2_s >> ic_1 >> ic_2 >> ic_3 >> i_s;
        if (infile.peek() == EOF)
            break;
        // cout << " ia1_s,ia2_s: " << ia1_s << "  " << ia2_s << endl;
        infile >> j_s >> mu_s;
        // cout<<ic_1<<mu_s<<endl;
        int ia1 = stoi(ia1_s) - 1;
        int ia2 = stoi(ia2_s) - 1;
        R[0] = stoi(ic_1);
        R[1] = stoi(ic_2);
        R[2] = stoi(ic_3);
        int n_i = stoi(i_s);
        int n_j = stoi(j_s);
        int n_mu = stoi(mu_s);

        if (std::find(ids.cbegin(), ids.cend(), id) != ids.cend())
        {
            shared_ptr<matrix> cs_ptr = make_shared<matrix>();
            cs_ptr->create(n_i * n_j, n_mu);

            for (int i = 0; i != n_i; i++)
                for (int j = 0; j != n_j; j++)
                    for (int mu = 0; mu != n_mu; mu++)
                    {
                        infile >> Cs_ele;
                        (*cs_ptr)(i * n_j + j, mu) = stod(Cs_ele);
                    }
            set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, cs_ptr->c, 0);
        }
        else
        {
            set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, nullptr, 1);

            double maxval = -1.0;
            for (int i = 0; i != n_i; i++)
                for (int j = 0; j != n_j; j++)
                    for (int mu = 0; mu != n_mu; mu++)
                    {
                        infile >> Cs_ele;
                        maxval = std::max(maxval, abs(stod(Cs_ele)));
                    }
            if (maxval < threshold) cs_discard++;
        }
        id++;
    }
    infile.close();
    return cs_discard;
}


static size_t handle_Cs_file_binary_by_ids(const string &file_path, double threshold, const vector<size_t> &ids)
{
    ifstream infile;
    int dims[8];
    int n_apcell_file;

    infile.open(file_path, std::ios::in | std::ios::binary);
    infile.read((char *) &natom, sizeof(int));
    infile.read((char *) &ncell, sizeof(int));
    infile.read((char *) &n_apcell_file, sizeof(int));
    size_t cs_discard = 0;

    int R[3];

    for (int i = 0; i < n_apcell_file; i++)
    {
        infile.read((char *) &dims[0], 8 * sizeof(int));
        // cout<<ic_1<<mu_s<<endl;
        int ia1 = dims[0] - 1;
        int ia2 = dims[1] - 1;
        R[0] = dims[2];
        R[1] = dims[3];
        R[2] = dims[4];
        int n_i = dims[5];
        int n_j = dims[6];
        int n_mu = dims[7];

        if (std::find(ids.cbegin(), ids.cend(), static_cast<size_t>(i)) != ids.cend())
        {
            shared_ptr<matrix> cs_ptr = make_shared<matrix>();
            cs_ptr->create(n_i * n_j, n_mu);
            infile.read((char *) cs_ptr->c, n_i * n_j * n_mu * sizeof(double));
            set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, cs_ptr->c, 0);
        }
        else
        {
            set_ao_basis_aux(ia1, ia2, n_i, n_j, n_mu, R, nullptr, 1);
            infile.seekg(n_i * n_j * n_mu * sizeof(double), ios::cur);
            cs_discard++;
        }
    }
    infile.close();
    return cs_discard;
}


size_t read_Cs_evenly_distribute(const string &dir_path, double threshold, int myid, int nprocs)
{
    size_t cs_discard = 0;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    vector<string> files;
    unordered_map<string, vector<size_t>> files_Cs_ids;
    unordered_map<string, vector<size_t>> files_Cs_ids_this_proc;
    bool binary;
    bool binary_checked = false;

    Profiler::start("handle_Cs_file_dry");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fn(ptr->d_name);
        if (fn.find("Cs_data") == 0)
        {
            files.push_back(dir_path + fn);
            if (!binary_checked)
            {
                binary = check_Cs_file_binary(dir_path + fn);
                binary_checked = true;
            }
        }
    }

    const auto nfiles = files.size();
    // cout << nfiles << "\n";

    // TODO: the IO can be improved, in two possible ways
    // 1. Each MPI task reads only a subset of files, instead of all files.
    // 2. Parallel reading for each file. This may be more efficient, but would be more difficult to implement
    for (int i_fn = 0; i_fn != nfiles; i_fn++)
    {
        // Let each MPI process read different files at one time
        auto i_fn_myid = (i_fn + myid * nfiles / nprocs) % files.size();
        const auto &fn = files[i_fn_myid];
        std::vector<size_t> ids_keep_this_file;
        if (binary)
        {
            ids_keep_this_file = handle_Cs_file_binary_dry(fn, threshold);
        }
        else
        {
            ids_keep_this_file = handle_Cs_file_dry(fn, threshold);
        }
        files_Cs_ids[fn] = ids_keep_this_file;
    }

    // Filter out the Cs to be actually read in each process
    size_t id_total = 0;
    for (int i_fn = 0; i_fn < nfiles; i_fn++)
    {
        const auto &fn = files[i_fn];
        const auto &ids_this_file = files_Cs_ids[fn];
        for (int id = 0; id != ids_this_file.size(); id++)
        {
            if (id_total % nprocs == myid) files_Cs_ids_this_proc[fn].push_back(ids_this_file[id]);
            id_total++;
        }
    }
    Profiler::stop("handle_Cs_file_dry");
    closedir(dir);
    dir = NULL;
    if (myid == 0) LIBRPA::utils::lib_printf("Finished Cs filtering\n");

    Profiler::start("handle_Cs_file");
    // cout << files_Cs_ids_this_proc.size() << "\n";
    // LIBRPA::envs::ofs_myid << files_Cs_ids_this_proc << "\n";
    for (const auto& fn_ids: files_Cs_ids_this_proc)
    {
        LIBRPA::envs::ofs_myid << fn_ids.first << " " << fn_ids.second << endl;
        if (binary)
        {
            cs_discard += handle_Cs_file_binary_by_ids(fn_ids.first, threshold, fn_ids.second);
        }
        else
        {
            cs_discard += handle_Cs_file_by_ids(fn_ids.first, threshold, fn_ids.second);
        }
    }
    Profiler::stop("handle_Cs_file");

    // initialize basis set object
    LIBRPA::atomic_basis_wfc.set(atom_nw);
    LIBRPA::atomic_basis_abf.set(atom_mu);
    
    atom_mu_part_range.resize(atom_mu.size());
    atom_mu_part_range[0]=0;
    for(int I=1;I!=atom_mu.size();I++)
        atom_mu_part_range[I]=atom_mu.at(I-1)+atom_mu_part_range[I-1];
    
    N_all_mu=atom_mu_part_range[natom-1]+atom_mu[natom-1];
    cout << "Done\n";
    return cs_discard;
}

void get_natom_ncell_from_first_Cs_file(int &n_atom, int &n_cell, const string &dir_path)
{
    // cout<<file_path<<endl;
    ifstream infile;
    bool binary;

    string file_path = "";

    // find Cs file
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    while ((ptr = readdir(dir)) != NULL)
    {
        string fn(ptr->d_name);
        if (fn.find("Cs_data") == 0)
        {
            file_path = dir_path + fn;
            break;
        }
    }
    if (file_path == "")
        throw std::runtime_error("Cs_data file is not found under dir_path: " + dir_path);

    binary = check_Cs_file_binary(file_path);
    if (LIBRPA::envs::myid_global == 0)
    {
        if (binary)
        {
            cout << "Unformatted binary Cs files detected" << endl;
        }
        else
        {
            cout << "ASCII format Cs files detected" << endl;
        }
    }

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *) &n_atom, sizeof(int));
        infile.read((char *) &n_cell, sizeof(int));
        infile.close();
    }
    else
    {
        string natom_s, ncell_s;
        infile.open(file_path);
        infile >> natom_s >> ncell_s;
        // cout<<"  natom_s:"<<natom_s<<"  ncell_s: "<<ncell_s<<endl;
        n_atom = stoi(natom_s);
        n_cell = stoi(ncell_s);
        infile.close();
    }
}

void read_dielec_func(const string &file_path, std::vector<double> &omegas, std::vector<double> &dielec_func_imagfreq)
{
    std::ifstream ifs;
    double omega, re, im;
    ifs.open(file_path);

    if (!ifs.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    while(ifs >> omega >> re >> im)
    {
        omegas.push_back(omega);
        dielec_func_imagfreq.push_back(re);
    }
    ifs.close();
}


static int handle_Vq_full_file(const string &file_path, map<Vector3_Order<double>, ComplexMatrix> &Vq_full, bool binary)
{
    // cout << "Begin to read aims vq_real from " << file_path << endl;
    ifstream infile;
    int n_irk_points_local;

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *) &n_irk_points, sizeof(int));
        infile.read((char *) &n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }

    if (!infile.good())
        return 1;

    if (binary)
    {
        int nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;

        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *) &nbasbas, sizeof(int));
            infile.read((char *) &brow, sizeof(int));
            infile.read((char *) &erow, sizeof(int));
            infile.read((char *) &bcol, sizeof(int));
            infile.read((char *) &ecol, sizeof(int));
            infile.read((char *) &iq, sizeof(int));
            infile.read((char *) &q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;
            Vector3_Order<double> qvec(kvec_c[iq]);

            if (irk_weight.count(qvec) == 0)
            {
                irk_points.push_back(qvec);
                irk_weight.insert(pair<Vector3_Order<double>, double>(qvec, q_weight));
            }
            if (!Vq_full.count(qvec))
            {
                Vq_full[qvec].create(nbasbas, nbasbas);
            }

            const int nrow = erow - brow + 1;
            const int ncol = ecol - bcol + 1;
            const size_t n = nrow * ncol;
            std::vector<complex<double>> tmp(n);
            infile.read((char *) tmp.data(), 2 * n * sizeof(double));
            for (int i = 0; i < nrow; i++)
            {
                for (int j = 0; j < ncol; j++)
                {
                    const auto i_mu = i + brow;
                    const auto i_nu = j + bcol;
                    Vq_full[qvec](i_mu, i_nu) = tmp[i * ncol + j]; // for abacus
                }
            }
        }
    }
    else
    {
        string nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i, q_num, q_weight;
        while (infile.peek() != EOF)
        {
            infile >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF)
                break;
            if (!infile.good())
                return 2;
            //cout << "vq range: " << begin_row << " ~ " << end_row << "  ,   " << begin_col << " ~ " << end_col << endl;
            infile >> q_num >> q_weight;
            if (!infile.good())
                return 3;
            int mu = stoi(nbasbas);
            int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;
            
            //skip empty coulumb_file
            if((erow-brow<=0) || (ecol-bcol<=0) || iq<0 || iq> klist.size())
                return 4;
            Vector3_Order<double> qvec(kvec_c[iq]);
            // skip duplicate insert of k weight, since 
            if (irk_weight.count(qvec) == 0)
            {
                irk_points.push_back(qvec);
                irk_weight.insert(pair<Vector3_Order<double>, double>(qvec, stod(q_weight)));
            }
            if (!Vq_full.count(qvec))
            {
                Vq_full[qvec].create(mu, nu);
            }
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    //Vq_full[qvec](i_nu, i_mu) = complex<double>(stod(vq_r), stod(vq_i)); // for FHI-aims
                    Vq_full[qvec](i_mu, i_nu) = complex<double>(stod(vq_r), stod(vq_i)); // for abacus
                }
            }
        }
    }
    return 0;
}

size_t read_Vq_full(const string &dir_path, const string &vq_fprefix, bool is_cut_coulomb)
{
    size_t vq_save = 0;
    size_t vq_discard = 0;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    vector<string> files;
    map<Vector3_Order<double>, ComplexMatrix> Vq_full;

    bool binary;
    bool binary_checked = false;

    Profiler::start("handle_Vq_full_file");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (fm.find(vq_fprefix) == 0)
        {
            string file_path = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_coulomb_file_binary(file_path);
                binary_checked = true;
                if (LIBRPA::envs::myid_global == 0)
                {
                    if (binary)
                    {
                        cout << "Unformatted binary V files detected" << endl;
                    }
                    else
                    {
                        cout << "ASCII format V files detected" << endl;
                    }
                }
            }

            int retcode = handle_Vq_full_file(file_path, Vq_full, binary);
            if (retcode != 0)
            {
                LIBRPA::utils::lib_printf("Error encountered when reading %s, return code %d", fm.c_str(), retcode);
            }
        }
    }
    Profiler::stop("handle_Vq_full_file");

    // cout << "FINISH coulomb files reading!" << endl;
    Profiler::start("set_aux_cut_coulomb_k_atom_pair_out");
    for (auto &vf_p : Vq_full)
    {
        auto qvec = vf_p.first;
        int iq = -1;
        auto ite_q = std::find(klist.cbegin(), klist.cend(), qvec);
        if (ite_q != klist.cend())
        {
            iq = std::distance(klist.cbegin(), ite_q);
        }
        else
        {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + ":" + std::string(__FUNCTION__) + ": "
                "fail to find qvec in klist, qvec = " + 
                std::to_string(qvec.x) + " " + std::to_string(qvec.y) + " " + std::to_string(qvec.z));
        }
        
        // cout << "Qvec:" << qvec << endl;
        for (int I = 0; I != atom_mu.size(); I++)
        {
            for (int J = 0; J != atom_mu.size(); J++)
            {
                // Coulomb is Hermitian, only parse upper half
                if (I > J)
                {
                    continue;
                }

                // Vq_full stores the full matrix, parse by I-J block
                // The matrices have to be duplicated ...
                matrix re(atom_mu[I], atom_mu[J]), im(atom_mu[I], atom_mu[J]);

                // vq_ptr_tran->create(atom_mu[J],atom_mu[I]);
                // cout << "I J: " << I << "  " << J << "   mu,nu: " << atom_mu[I] << "  " << atom_mu[J] << endl;
                for (int i_mu = 0; i_mu != atom_mu[I]; i_mu++)
                {
                    for (int i_nu = 0; i_nu != atom_mu[J]; i_nu++)
                    {
                        //(*vq_ptr)(i_mu, i_nu) = vf_p.second(atom_mu_loc2glo(J, i_nu), atom_mu_loc2glo(I, i_mu)); ////for aims
                        re(i_mu, i_nu) = vf_p.second(atom_mu_loc2glo(I, i_mu), atom_mu_loc2glo(J, i_nu)).real(); // for abacus
                        im(i_mu, i_nu) = vf_p.second(atom_mu_loc2glo(I, i_mu), atom_mu_loc2glo(J, i_nu)).imag();
                    }
                }

                if (is_cut_coulomb)
                {
                    set_aux_cut_coulomb_k_atom_pair(iq, I, J, atom_mu[I], atom_mu[J], re.c, im.c);
                }
                else
                {
                    set_aux_bare_coulomb_k_atom_pair(iq, I, J, atom_mu[I], atom_mu[J], re.c, im.c);
                }
                // if (I == J)
                // {
                //     (*vq_ptr).set_as_identity_matrix();
                // }

                // if ((*vq_ptr).real().absmax() >= threshold)
                // {
                //     coulomb_mat[I][J][qvec] = vq_ptr;
                //     vq_save++;
                // }
                // else
                // {
                //     vq_discard++;
                // }
            }
        }
    }
    Profiler::stop("set_aux_cut_coulomb_k_atom_pair_out");
    closedir(dir);
    dir = NULL;
    // cout << "vq threshold: " << threshold << endl;
    // cout << "vq_save:    " << vq_save << endl;
    // cout << "vq_dicard:  " << vq_discard << endl;
    // cout << "  Vq_dim   " << coulomb_mat.size() << "    " << coulomb_mat[0].size() << "   " << coulomb_mat[0][0].size() << endl;
    // for (auto &irk : irk_weight)
    // {
    //     cout << " irk_vec and weight: " << irk.first << "  " << irk.second << endl;
    // }
    // cout << "Finish read aims vq" << endl;
    return vq_discard;
}


static int handle_Vq_row_file(const string &file_path, double threshold,
        atom_mapping<std::map<int, std::shared_ptr<ComplexMatrix>>>::pair_t_old &coulomb,
        const vector<atpair_t> &local_atpair, bool binary)
{
    // cout << "Begin to read aims vq_real from " << file_path << endl;
    ifstream infile;
    int n_irk_points_local;

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *) &n_irk_points, sizeof(int));
        infile.read((char *) &n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }
    if (!infile.good()) return 1;

    if (binary)
    {
        set<int> coulomb_row_need;
        for (const auto &ap: local_atpair)
        {
            const auto brow = atom_mu_part_range[ap.first];
            const auto nb = atom_mu[ap.first];
            for (int ir = 0; ir < nb; ir++)
            {
                coulomb_row_need.insert(brow + ir);
            }
        }

        int nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;
        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *) &nbasbas, sizeof(int));
            infile.read((char *) &brow, sizeof(int));
            infile.read((char *) &erow, sizeof(int));
            infile.read((char *) &bcol, sizeof(int));
            infile.read((char *) &ecol, sizeof(int));
            infile.read((char *) &iq, sizeof(int));
            infile.read((char *) &q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;

            Vector3_Order<double> qvec(kvec_c[iq]);
            if (irk_weight.count(qvec) == 0)
            {
                irk_points.push_back(qvec);
                irk_weight.insert(pair<Vector3_Order<double>, double>(qvec, q_weight));
            }

            for (const auto &ap : local_atpair)
            {
                auto I = ap.first;
                auto J = ap.second;
                if (coulomb[I][J].count(iq) == 0)
                {
                    shared_ptr<ComplexMatrix> vq_ptr = make_shared<ComplexMatrix>();
                    vq_ptr->create(atom_mu[I], atom_mu[J]);
                    coulomb[I][J][iq] = vq_ptr;
                }
            }

            const auto ncol = ecol - bcol + 1;

            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                vector<complex<double>> tmp_row(ncol);
                infile.read((char *) tmp_row.data(), 2 * ncol * sizeof(double));

                if (coulomb_row_need.count(i_mu))
                {
                    int I_loc, mu_loc;
                    I_loc = atom_mu_glo2loc(i_mu, mu_loc);
                    for (auto &Jp : coulomb[I_loc])
                    {
                        auto J = Jp.first;
                        int Jb = atom_mu_part_range[J];
                        int Je = atom_mu_part_range[J] + atom_mu[J] - 1;

                        if (ecol >= Jb && bcol < Je)
                        {
                            int start_point = (bcol <= Jb ? Jb : bcol);
                            int end_point = (ecol <= Je ? ecol : Je);
                            for (int i = start_point; i <= end_point; i++)
                            {
                                int J_loc, nu_loc;
                                J_loc = atom_mu_glo2loc(i, nu_loc);
                                // printf("|i: %d   J: %d   J_loc: %d, nu_loc:
                                // %d\n",i,J,J_loc,nu_loc);
                                assert(J == J_loc);
                                (*coulomb[I_loc][J_loc][iq])(mu_loc, nu_loc) = tmp_row[i - bcol];
                            }
                        }
                    }
                }
            }
        }

    }
    else
    {
        string nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i, q_num, q_weight;
        while (infile.peek() != EOF)
        {
            infile >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF)
                break;
            if (!infile.good()) return 2;
            // cout << "vq range: " << begin_row << " ~ " << end_row << "  ,   " << begin_col << " ~ " << end_col << endl;
            infile >> q_num >> q_weight;
            if (!infile.good()) return 3;
            int mu = stoi(nbasbas);
            int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;
            //cout<<file_path<<" iq:"<<iq<<"  qweight:"<<stod(q_weight)<<endl;

            //skip empty coulumb_file
            if((erow-brow<=0) || (ecol-bcol<=0) || iq<0 || iq> klist.size())
                return 4;

            Vector3_Order<double> qvec(kvec_c[iq]);
            // skip duplicate insert of k weight, since 
            if (irk_weight.count(qvec) == 0)
            {
                irk_points.push_back(qvec);
                irk_weight.insert(pair<Vector3_Order<double>, double>(qvec, stod(q_weight)));
            }

            for(const auto &ap:local_atpair)
            {
                auto I=ap.first;
                auto J=ap.second;
                if(!coulomb[I][J].count(iq))
                {
                    shared_ptr<ComplexMatrix> vq_ptr = make_shared<ComplexMatrix>();
                    vq_ptr->create(atom_mu[I], atom_mu[J]);
                    // cout<<"  create  IJ: "<<I<<"  "<<J<<"   "<<atom_mu[I]<<"  "<<atom_mu[J];
                    coulomb[I][J][iq]=vq_ptr;
                }
            }   

            set<int> coulomb_row_need;
            for(auto &Ip:coulomb)
                for(int ir=atom_mu_part_range[Ip.first];ir!=atom_mu_part_range[Ip.first]+atom_mu[Ip.first];ir++)
                    coulomb_row_need.insert(ir);

            //printf("   |process %d, coulomb_begin:  %d, size: %d\n",para_mpi.get_myid(),*coulomb_row_need.begin(),coulomb_row_need.size());
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                vector<complex<double>> tmp_row(ecol-bcol+1);
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    if (!infile.good()) return 4;
                    
                    tmp_row[i_nu-bcol] = complex<double>(stod(vq_r), stod(vq_i)); // for abacus
                    
                }
                if(coulomb_row_need.count(i_mu))
                {
                    int I_loc,mu_loc;
                    I_loc=atom_mu_glo2loc(i_mu,mu_loc);
                    int bI=atom_mu_part_range[I_loc];
                    for(auto &Jp:coulomb[I_loc] )
                    {
                        auto J=Jp.first;
                        int Jb=atom_mu_part_range[J];
                        int Je=atom_mu_part_range[J]+atom_mu[J]-1;
                        
                        if(ecol>=Jb && bcol<Je)
                        {
                            int start_point = ( bcol<=Jb ? Jb:bcol);
                            int end_point = (ecol<=Je? ecol:Je);
                            for(int i=start_point;i<=end_point;i++)
                            {
                                int J_loc, nu_loc;
                                J_loc=atom_mu_glo2loc(i,nu_loc);
                                //printf("|i: %d   J: %d   J_loc: %d, nu_loc: %d\n",i,J,J_loc,nu_loc);
                                assert(J==J_loc);
                                (*coulomb[I_loc][J_loc][iq])(mu_loc,nu_loc)=tmp_row[i-bcol];
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}


size_t read_Vq_row(const string &dir_path, const string &vq_fprefix, double threshold,
        const vector<atpair_t> &local_atpair, bool is_cut_coulomb)
{
    cout<<"Begin READ_Vq_Row"<<endl;
    set<int> local_I_set;
    for(auto &lap:local_atpair)
    {
        local_I_set.insert(lap.first);
        local_I_set.insert(lap.second);
    }

    size_t vq_save = 0;
    size_t vq_discard = 0;
    atom_mapping<std::map<int, std::shared_ptr<ComplexMatrix>>>::pair_t_old coulomb;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    vector<string> files;
    bool binary;
    bool binary_checked = false;

    //map<Vector3_Order<double>, ComplexMatrix> Vq_full;
    Profiler::start("handle_Vq_row_file");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (fm.find(vq_fprefix) == 0)
        {
            string file_path = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_coulomb_file_binary(file_path);
                binary_checked = true;
                if (LIBRPA::envs::myid_global == 0)
                {
                    if (binary)
                    {
                        cout << "Unformatted binary V files detected" << endl;
                    }
                    else
                    {
                        cout << "ASCII format V files detected" << endl;
                    }
                }
            }
            handle_Vq_row_file(file_path, threshold, coulomb, local_atpair, binary);
        }
    }
    Profiler::stop("handle_Vq_row_file");

    // MYZ: now the map coulomb contains the complete atom-pair matrix.
    // Call the API to parse the data.
    // To reduce memory consumption during this process, we erase the data in temporary object once it is parsed.
    auto it_I = coulomb.begin();
    Profiler::start("set_aux_coulomb_k_atom_pair");
    while (it_I != coulomb.end())
    {
        auto I = it_I->first;
        auto it_J = it_I->second.begin();
        while (it_J != it_I->second.end())
        {
            auto J = it_J->first;
            auto it_iq = it_J->second.begin();
            while (it_iq != it_J->second.end())
            {
                auto iq = it_iq->first;
                auto &vq_ptr = it_iq->second;
                if (is_cut_coulomb)
                {
                    set_aux_cut_coulomb_k_atom_pair(iq, I, J, vq_ptr->nr, vq_ptr->nc, vq_ptr->real().c, vq_ptr->imag().c);
                }
                else
                {
                    set_aux_bare_coulomb_k_atom_pair(iq, I, J, vq_ptr->nr, vq_ptr->nc, vq_ptr->real().c, vq_ptr->imag().c);
                }
                it_iq = it_J->second.erase(it_iq);
            }
            it_J = it_I->second.erase(it_J);
        }
        it_I = coulomb.erase(it_I);
    }
    Profiler::stop("set_aux_coulomb_k_atom_pair");

    // cout << "FINISH coulomb files reading!" << endl;

    closedir(dir);
    dir = NULL;

    // ofstream fs;
    // std::stringstream ss;
    // ss<<"out_coulomb_rank_"<<para_mpi.get_myid()<<".txt";
    // fs.open(ss.str());
    // for(auto &Ip:coulomb_mat)
    // {
    //     for(auto &Jp:Ip.second)
    //         for(auto &qp:Jp.second)
    //         {
    //             std::stringstream sm;
    //             sm<<"I,J "<<Ip.first<<"  "<<Jp.first;
    //             //printf("|process %d  I J: %d, %d\n",para_mpi.get_myid(), Ip.first,Jp.first);
    //             print_complex_matrix_file(sm.str().c_str(),(*qp.second),fs,false);
    //         }
                
    // }
    // fs.close();
    return vq_discard;
}


void erase_Cs_from_local_atp(atpair_R_mat_t &Cs, vector<atpair_t> &local_atpair)
{
    //erase no need Cs
    
    set<size_t> loc_atp_index;
    for(auto &lap:local_atpair)
    {
        loc_atp_index.insert(lap.first);
        loc_atp_index.insert(lap.second);
    }
    vector<atom_t> Cs_first;
    for (const auto &Ip: Cs)
        Cs_first.push_back(Ip.first);
    for (const auto &I: Cs_first)
    {
        if(!loc_atp_index.count(I))
            Cs.erase(I);
    }
    // for(auto &Ip:Cs)
    //     if(!loc_atp_index.count(Ip.first))
    //     {
    //         Cs.erase(Ip.first);
    //     }
    LIBRPA::utils::release_free_mem();
    LIBRPA::utils::lib_printf("| process %d, size of Cs after erase: %lu\n", LIBRPA::envs::mpi_comm_global_h.myid, Cs.size());
}

void read_stru(const int& n_kpoints, const std::string &file_path)
{
    // cout << "Begin to read aims stru" << endl;
    ifstream infile;
    string x, y, z, tmp;
    infile.open(file_path);

    std::vector<double> lat_mat(9);
    std::vector<double> G_mat(9);

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        lat_mat[i * 3] = stod(x);
        lat_mat[i * 3 + 1] = stod(y);
        lat_mat[i * 3 + 2] = stod(z);
    }

    for (int i = 0; i < 3; i++)
    {
        infile >> x >> y >> z;
        G_mat[i * 3] = stod(x);
        G_mat[i * 3 + 1] = stod(y);
        G_mat[i * 3 + 2] = stod(z);
    }

    set_latvec_and_G(lat_mat.data(), G_mat.data());

    // Read coordinates of atoms
    // Read number of atoms
    // FIXME: move to LibRPA API
    int n_atom;
    coord.clear();
    coord_frac.clear();
    infile >> n_atom;
    for (int i = 0; i < n_atom; i++)
    {
        infile >> x >> y >> z >> tmp;
        coord[i] = {stod(x), stod(y), stod(z)};
        // convert to fractional using the reciprocal lattice vectors
        coord_frac[i] = {
            (coord[i][0] * G_mat[0] + coord[i][1] * G_mat[1] + coord[i][2] * G_mat[2]) / TWO_PI,
            (coord[i][0] * G_mat[3] + coord[i][1] * G_mat[4] + coord[i][2] * G_mat[5]) / TWO_PI,
            (coord[i][0] * G_mat[6] + coord[i][1] * G_mat[7] + coord[i][2] * G_mat[8]) / TWO_PI,
        };
    }

    // G.print();
    // Matrix3 latG = latvec * G.Transpose();
    // cout << " lat * G^T" << endl;
    // latG.print();

    int nk[3];
    for (int i = 0; i < 3; i++)
    {
        infile >> x;
        nk[i] = stoi(x);
    }
    assert(n_kpoints == nk[0] * nk[1] * nk[2]);
    std::vector<double> kvecs(3 * n_kpoints);
    // kvec_c = new Vector3<double>[n_kpoints];
    for (int i = 0; i != 3 * n_kpoints; i++)
    {
        infile >> x;
        kvecs[i] = stod(x);
    }
    set_kgrids_kvec_tot(nk[0], nk[1], nk[2], kvecs.data());

    // TODO: use API for IBZ mapping
    for (int i = 0; i != n_kpoints; i++)
    {
        infile >> x;
        int id_irk = stoi(x) - 1;
        irk_point_id_mapping.push_back(id_irk);
        map_irk_ks[klist[id_irk]].push_back(klist[i]);
    }
}


std::vector<Vector3_Order<double>> read_band_kpath_info(const string &file_path, int &n_basis, int &n_states, int &n_spin)
{
    std::vector<Vector3_Order<double>> kfrac_band;

    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string x, y, z;
    int n_kpoints_band;

    // Read dimensions in the first row
    infile >> x;
    n_basis = stoi(x);
    infile >> x;
    n_states = stoi(x);
    infile >> x;
    n_spin = stoi(x);
    infile >> x;
    n_kpoints_band = stoi(x);

    for (int i = 0; i < n_kpoints_band; i++)
    {
        infile >> x >> y >> z;
        kfrac_band.push_back({stod(x), stod(y), stod(z)});
    }

    infile.close();

    return kfrac_band;
}

MeanField read_meanfield_band(const string &dir_path, int n_basis, int n_states, int n_spin, int n_kpoints_band)
{
    MeanField mf_band(n_spin, n_kpoints_band, n_states, n_basis);
    std::string s1, s2, s3, s4, s5;

    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        // Load occupation weights and eigenvalues
        std::stringstream ss;
        ss << dir_path << "band_KS_eigenvalue_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";
        ifstream infile;
        infile.open(ss.str());

        for (int i_spin = 0; i_spin < n_spin; i_spin++)
        {
            for (int i_state = 0; i_state < n_states; i_state++)
            {
                infile >> s1 >> s2 >> s3 >> s4 >> s5;
                mf_band.get_weight()[i_spin](ik, i_state) = stod(s3);
                mf_band.get_eigenvals()[i_spin](ik, i_state) = stod(s4);
            }
        }

        infile.close();

        // Load eigenvectors
        ss.str("");
        ss.clear();
        ss << dir_path << "band_KS_eigenvector_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";
        infile.open(ss.str(), std::ios::in | std::ios::binary);

        for (int i_spin = 0; i_spin < n_spin; i_spin++)
        {
            const size_t nbytes = n_basis * n_states * sizeof(std::complex<double>);
            infile.read((char *) mf_band.get_eigenvectors()[i_spin][ik].c, nbytes);
        }

        infile.close();
    }

    // TODO: Fermi energy is not set

    return mf_band;
}

std::vector<matrix> read_vxc_band(const string &dir_path, int n_states, int n_spin, int n_kpoints_band)
{
    std::vector<matrix> vxc_band(n_spin);
    for (int i_spin = 0; i_spin < n_spin; i_spin++)
    {
        vxc_band[i_spin].create(n_kpoints_band, n_states);
    }
    std::string s1, s2, s3;

    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        // Load occupation weights and eigenvalues
        std::stringstream ss;
        ss << dir_path << "band_vxc_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";
        ifstream infile;
        infile.open(ss.str());
        ss.clear();

        for (int i_spin = 0; i_spin < n_spin; i_spin++)
        {
            for (int i_state = 0; i_state < n_states; i_state++)
            {
                infile >> s1 >> s2 >> s3;
                vxc_band[i_spin](ik, i_state) = stod(s3);
            }
        }

        infile.close();
    }
    return vxc_band;
}
