#!/usr/bin/env python3
"""Add an env-gated, read-only component dump to frozen exact847 Scheme A."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SCHEMA = "librpa-exact847-component-dump-instrumentation-v1"
EXPECTED_SOURCE_SHA256 = (
    "34e5c93fe12259f4838469b0e19b2c2316d4b6871ba9d6c9da5b3c057a01ab34"
)
ANCHOR = """                auto H0_GW_all = construct_H0_GW_cut(
                    meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all,
                    n_spins, n_kpoints, n_bands, qsgw_band0_unoccupied_keep,
                    qsgw_band0_cut_mode, qsgw_band0_cut_shift_ha);
"""
INSERTION = r'''

                const char *component_dump_env =
                    std::getenv("LIBRPA_QSGW_LEGACY_COMPONENT_DUMP");
                if (component_dump_env != nullptr
                    && std::string(component_dump_env) == "1")
                {
                    std::ostringstream component_dir_stream;
                    component_dir_stream << Params::output_dir
                                         << "qsgw_legacy_components/iter_"
                                         << std::setw(5) << std::setfill('0')
                                         << iteration << "/";
                    const std::string component_dir = component_dir_stream.str();
                    ensure_dir_band(component_dir);

                    std::ofstream metadata(component_dir + "metadata.txt");
                    if (!metadata.good())
                    {
                        throw std::runtime_error(
                            "Failed to open exact847 component metadata for write: "
                            + component_dir);
                    }
                    metadata << "schema librpa-exact847-component-dump-v1\n";
                    metadata << "iteration " << iteration << "\n";
                    metadata << "n_spins " << n_spins << "\n";
                    metadata << "n_kpoints " << n_kpoints << "\n";
                    metadata << "n_bands " << n_bands << "\n";
                    metadata << "n_frequencies "
                             << chi0.tfg.get_freq_nodes().size() << "\n";
                    int metadata_frequency_index = 0;
                    for (const auto &frequency : chi0.tfg.get_freq_nodes())
                    {
                        metadata << "frequency_ha " << metadata_frequency_index
                                 << " " << std::setprecision(17) << frequency
                                 << "\n";
                        ++metadata_frequency_index;
                    }

                    const auto component_path =
                        [&component_dir](const std::string &component,
                                         const int spin, const int kpoint) {
                            std::ostringstream path;
                            path << component_dir << component << "_spin_"
                                 << std::setw(2) << std::setfill('0')
                                 << spin + 1 << "_k_" << std::setw(6)
                                 << std::setfill('0') << kpoint + 1 << ".bin";
                            return path.str();
                        };
                    const auto dump_static_component =
                        [&component_path](
                            const std::string &component,
                            const std::map<int, std::map<int, Matz>> &values) {
                            for (const auto &spin_entry : values)
                            {
                                for (const auto &k_entry : spin_entry.second)
                                {
                                    write_matz_binary_band(
                                        k_entry.second,
                                        component_path(component,
                                                       spin_entry.first,
                                                       k_entry.first));
                                }
                            }
                        };

                    dump_static_component("h0", H_KS0);
                    dump_static_component("vxc_dft", vxc0);
                    dump_static_component("exx", exx.exx_is_ik_KS);
                    dump_static_component("vc", Vc_all);
                    dump_static_component("raw_h", H0_GW_all);

                    int frequency_index = 0;
                    for (const auto &frequency : chi0.tfg.get_freq_nodes())
                    {
                        for (int spin = 0; spin < n_spins; ++spin)
                        {
                            for (int kpoint = 0; kpoint < n_kpoints; ++kpoint)
                            {
                                std::ostringstream component;
                                component << "sigma_c_iw_" << std::setw(3)
                                          << std::setfill('0')
                                          << frequency_index;
                                write_matz_binary_band(
                                    s_g0w0.sigc_is_ik_f_KS.at(spin)
                                        .at(kpoint).at(frequency),
                                    component_path(component.str(), spin,
                                                   kpoint));
                            }
                        }
                        ++frequency_index;
                    }
                }
'''


class InstrumentationError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def instrument(source: bytes) -> bytes:
    observed = sha256_bytes(source)
    if observed != EXPECTED_SOURCE_SHA256:
        raise InstrumentationError(
            f"source sha256 mismatch: {observed} != {EXPECTED_SOURCE_SHA256}"
        )
    text = source.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise InstrumentationError("expected exactly one Hamiltonian anchor")
    if "LIBRPA_QSGW_LEGACY_COMPONENT_DUMP" in text:
        raise InstrumentationError("component dump instrumentation already present")
    return text.replace(ANCHOR, ANCHOR + INSERTION, 1).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    output = instrument(source)
    args.output.write_bytes(output)
    print(f"schema={SCHEMA}")
    print(f"source_sha256={sha256_bytes(source)}")
    print(f"output_sha256={sha256_bytes(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
