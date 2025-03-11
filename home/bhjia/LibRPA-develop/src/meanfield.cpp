void MeanField::broadcast(const LIBRPA::MPI_COMM_handler& comm_hdl, int root) {
    // 广播基本类型成员
    comm_hdl.bcast(n_spins, root);
    comm_hdl.bcast(n_aos, root);
    comm_hdl.bcast(n_bands, root);
    comm_hdl.bcast(n_kpoints, root);
    comm_hdl.bcast(efermi, root);

    // 广播矩阵数据
    auto bcast_matrix = [&](matrix& m) {
        comm_hdl.bcast(m.nr, root);
        comm_hdl.bcast(m.nc, root);
        if(comm_hdl.rank() != root) m.create(m.nr, m.nc);
        comm_hdl.bcast(m.data, m.size(), root);
    };

    // 广播vector<matrix>
    auto bcast_matrix_vec = [&](std::vector<matrix>& vec) {
        size_t vec_size = vec.size();
        comm_hdl.bcast(vec_size, root);
        if(comm_hdl.rank() != root) vec.resize(vec_size);
        for(auto& m : vec) bcast_matrix(m);
    };

    // 广播主要数据成员
    bcast_matrix_vec(eskb);
    bcast_matrix_vec(wg);
    bcast_matrix_vec(wg0);

    // 广播波函数数据
    auto bcast_complex_matrix = [&](ComplexMatrix& cm) {
        comm_hdl.bcast(cm.nr, root);
        comm_hdl.bcast(cm.nc, root);
        if(comm_hdl.rank() != root) cm.create(cm.nr, cm.nc);
        comm_hdl.bcast(cm.data, cm.size(), root);
    };

    // 广播wfc和wfc0
    auto bcast_wfc = [&](std::vector<std::vector<ComplexMatrix>>& wfc_vec) {
        comm_hdl.bcast(wfc_vec.size(), root);
        for(auto& vec : wfc_vec) {
            comm_hdl.bcast(vec.size(), root);
            for(auto& cm : vec) bcast_complex_matrix(cm);
        }
    };

    if(comm_hdl.rank() != root) {
        wfc.resize(n_spins);
        wfc0.resize(n_spins);
    }
    bcast_wfc(wfc);
    bcast_wfc(wfc0);
}
