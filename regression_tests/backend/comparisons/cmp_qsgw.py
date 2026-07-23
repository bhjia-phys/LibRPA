import math
import re


__all__ = [
    "band_iterations",
    "matrix_trace",
    "eigenvalue_trace",
    "iteration_summary",
]


HA2EV = 27.211386245988

MATRIX_HEADER = (
    "# iter channel component spin kpoint frequency_index frequency_Ha "
    "row column real_value imag_value"
)
EIGENVALUE_HEADER = (
    "# iter channel spin kpoint kx ky kz band energy_eV"
)
SUMMARY_HEADER = (
    "# iter max_delta_eV residual_l2_Ha residual_max_Ha "
    "efermi_eV gap_eV electron_count requested_mode applied_mode beta "
    "fallback rcond coefficient_l1 coefficient_count converged "
    "coefficients fallback_reason"
)

CONTRACT_KEYS = frozenset((
    "qsgw_contract_version",
    "fixed_basis",
    "live_update",
    "velocity",
    "head",
    "wing",
    "headwing",
    "symmetry",
    "hartree",
    "hartree_coulomb",
    "hartree_normalization",
    "band",
    "h_qsgw_cut",
    "qsgw_band0_unoccupied_keep",
    "qsgw_band0_cut_mode",
    "qsgw_band0_cut_shift_ha",
    "qsgw_input_contract",
    "qsgw_input_contract_sha256",
    "qsgw_mixer",
    "qsgw_mixing_beta",
))
REQUIRED_CONTRACT_KEYS = frozenset((
    "qsgw_contract_version",
    "fixed_basis",
    "live_update",
    "velocity",
    "symmetry",
    "hartree",
    "band",
    "qsgw_input_contract",
    "qsgw_input_contract_sha256",
    "qsgw_mixer",
    "qsgw_mixing_beta",
))

HERMITIAN_COMPONENTS = frozenset((
    "h0",
    "vxc_dft",
    "exx",
    "vc",
    "delta_vh",
    "raw_h",
    "mixed_h",
    "projected_h",
))


def band_iterations(occupied_bands, energy_tolerance_ev="1e-4",
                    gap_tolerance_ev="2e-4",
                    coordinate_tolerance="1e-7", precision="3"):
    """Compare every QSGW band energy and indirect gap in each iteration."""
    occupied_bands = int(occupied_bands)
    if occupied_bands <= 0:
        raise ValueError("occupied_bands must be positive")
    energy_tolerance_ev = _positive_float(
        energy_tolerance_ev, "energy_tolerance_ev", allow_zero=True)
    gap_tolerance_ev = _positive_float(
        gap_tolerance_ev, "gap_tolerance_ev", allow_zero=True)
    coordinate_tolerance = _positive_float(
        coordinate_tolerance, "coordinate_tolerance", allow_zero=True)
    precision = int(precision)

    def inner(test_files, reference_files):
        try:
            pairs = _paired_file_texts(test_files, reference_files)
            trajectories = {}
            maximum_coordinate = 0.0
            maximum_energy = 0.0
            maximum_gap = 0.0
            maximum_location = None

            for filename, test_text, reference_text in pairs:
                spin, iteration = _parse_band_filename(filename)
                trajectories.setdefault(spin, []).append(iteration)
                test_rows = _parse_band_table(
                    test_text, "test {}".format(filename))
                reference_rows = _parse_band_table(
                    reference_text, "reference {}".format(filename))
                if len(test_rows) != len(reference_rows):
                    return False, (
                        "{}: k-point count mismatch: {} != {}"
                        .format(filename, len(test_rows), len(reference_rows))
                    )
                test_bands = len(test_rows[0][1])
                reference_bands = len(reference_rows[0][1])
                if test_bands != reference_bands:
                    return False, (
                        "{}: band count mismatch: {} != {}"
                        .format(filename, test_bands, reference_bands)
                    )
                if occupied_bands >= reference_bands:
                    return False, (
                        "{}: occupied band count {} is outside 1..{}"
                        .format(filename, occupied_bands,
                                reference_bands - 1)
                    )

                for kpoint, (test_row, reference_row) in enumerate(
                        zip(test_rows, reference_rows)):
                    coordinate_difference = max(
                        abs(left - right)
                        for left, right in zip(
                            test_row[0], reference_row[0])
                    )
                    maximum_coordinate = max(
                        maximum_coordinate, coordinate_difference)
                    for band, (test_energy, reference_energy) in enumerate(
                            zip(test_row[1], reference_row[1])):
                        difference = abs(test_energy - reference_energy)
                        if difference > maximum_energy:
                            maximum_energy = difference
                            maximum_location = (
                                filename, kpoint + 1, band + 1)

                test_gap = _band_gap(test_rows, occupied_bands)
                reference_gap = _band_gap(reference_rows, occupied_bands)
                maximum_gap = max(
                    maximum_gap, abs(test_gap - reference_gap))

            niterations = _validate_band_trajectories(trajectories)
            message = (
                "max abs band-energy diff = {:.{p}E} eV "
                "(tol = {:.{p}E} eV), max gap diff = {:.{p}E} eV "
                "(tol = {:.{p}E} eV), max k-point coordinate diff = "
                "{:.{p}E} over {} iterations and {} spins"
            ).format(
                maximum_energy, energy_tolerance_ev,
                maximum_gap, gap_tolerance_ev,
                maximum_coordinate, niterations, len(trajectories),
                p=precision)
            if maximum_location is not None:
                message += ", max at {} k-point {} band {}".format(
                    *maximum_location)
            passed = (
                maximum_energy <= energy_tolerance_ev
                and maximum_gap <= gap_tolerance_ev
                and maximum_coordinate <= coordinate_tolerance
            )
            return passed, message
        except (TypeError, ValueError) as error:
            return False, str(error)

    return inner


def matrix_trace(relative_tolerance="1e-8", absolute_tolerance="1e-12",
                 hermiticity_tolerance="1e-10",
                 unitarity_tolerance="1e-10",
                 require_complete_trajectory="true", precision="3"):
    """Compare keyed QSGW matrix traces and enforce matrix invariants."""
    relative_tolerance = _positive_float(
        relative_tolerance, "relative_tolerance")
    absolute_tolerance = _positive_float(
        absolute_tolerance, "absolute_tolerance", allow_zero=True)
    hermiticity_tolerance = _positive_float(
        hermiticity_tolerance, "hermiticity_tolerance", allow_zero=True)
    unitarity_tolerance = _positive_float(
        unitarity_tolerance, "unitarity_tolerance", allow_zero=True)
    require_complete_trajectory = _as_bool(
        require_complete_trajectory, "require_complete_trajectory")
    precision = int(precision)

    def inner(test_files, reference_files):
        try:
            pairs = _paired_file_texts(test_files, reference_files)
            nblocks = 0
            maximum_relative = 0.0
            maximum_absolute = 0.0
            maximum_hermiticity = 0.0
            maximum_unitarity = 0.0
            maximum_location = None

            for filename, test_text, reference_text in pairs:
                contract = _require_same_contract(
                    test_text, reference_text, filename)
                test_blocks = _parse_matrix_trace(
                    test_text, "test {}".format(filename))
                reference_blocks = _parse_matrix_trace(
                    reference_text, "reference {}".format(filename))
                if require_complete_trajectory:
                    _validate_matrix_trajectory(
                        test_blocks, contract, "test {}".format(filename))
                    _validate_matrix_trajectory(
                        reference_blocks, contract,
                        "reference {}".format(filename))
                mismatch = _key_mismatch(test_blocks, reference_blocks)
                if mismatch is not None:
                    return False, "{}: matrix block key set mismatch; {}".format(
                        filename, mismatch)

                for block_key in sorted(test_blocks, key=str):
                    test_frequency, test_matrix = test_blocks[block_key]
                    reference_frequency, reference_matrix = \
                        reference_blocks[block_key]
                    if abs(test_frequency - reference_frequency) > absolute_tolerance:
                        return False, (
                            "{}: frequency mismatch for block {}: {:.6E} > {:.6E}"
                            .format(filename, block_key,
                                    abs(test_frequency - reference_frequency),
                                    absolute_tolerance)
                        )
                    if _shape(test_matrix) != _shape(reference_matrix):
                        return False, "{}: shape mismatch for block {}".format(
                            filename, block_key)

                    difference_norm = _difference_frobenius(
                        test_matrix, reference_matrix)
                    reference_norm = _frobenius(reference_matrix)
                    if reference_norm > absolute_tolerance:
                        relative = difference_norm / reference_norm
                        within_tolerance = relative <= relative_tolerance
                        allowed = relative_tolerance * reference_norm
                    else:
                        relative = (0.0 if difference_norm <= absolute_tolerance
                                    else math.inf)
                        within_tolerance = difference_norm <= absolute_tolerance
                        allowed = absolute_tolerance
                    if not within_tolerance:
                        return False, (
                            "{}: Frobenius tolerance exceeded for block {}: "
                            "abs={:.6E}, rel={:.6E}, allowed={:.6E}"
                            .format(filename, block_key, difference_norm,
                                    relative, allowed)
                        )
                    if relative > maximum_relative:
                        maximum_relative = relative
                        maximum_location = (filename, block_key)
                    maximum_absolute = max(maximum_absolute, difference_norm)

                    component = block_key[2]
                    if component in HERMITIAN_COMPONENTS:
                        for side, matrix in (("test", test_matrix),
                                             ("reference", reference_matrix)):
                            residual = _hermiticity_residual(matrix)
                            maximum_hermiticity = max(
                                maximum_hermiticity, residual)
                            if residual > hermiticity_tolerance:
                                return False, (
                                    "{}: {} Hermiticity residual for block {} "
                                    "is {:.6E} (tol={:.6E})"
                                    .format(filename, side, block_key, residual,
                                            hermiticity_tolerance)
                                )
                    if component == "rotation_u":
                        for side, matrix in (("test", test_matrix),
                                             ("reference", reference_matrix)):
                            residual = _unitarity_residual(matrix)
                            maximum_unitarity = max(
                                maximum_unitarity, residual)
                            if residual > unitarity_tolerance:
                                return False, (
                                    "{}: {} unitarity residual for block {} "
                                    "is {:.6E} (tol={:.6E})"
                                    .format(filename, side, block_key, residual,
                                            unitarity_tolerance)
                                )
                    nblocks += 1

            message = (
                "max relative Frobenius = {:.{p}E}, max absolute Frobenius = "
                "{:.{p}E}, max Hermiticity residual = {:.{p}E}, max "
                "unitarity residual = {:.{p}E} over {} blocks"
            ).format(maximum_relative, maximum_absolute,
                     maximum_hermiticity, maximum_unitarity, nblocks,
                     p=precision)
            if maximum_location is not None:
                message += ", max at {} block {}".format(*maximum_location)
            return True, message
        except (TypeError, ValueError) as error:
            return False, str(error)

    return inner


def eigenvalue_trace(tolerance_ha="1e-6", coordinate_tolerance="1e-12",
                     precision="3"):
    """Compare keyed QSGW eigenvalue traces using a Hartree tolerance."""
    tolerance_ha = _positive_float(tolerance_ha, "tolerance_ha",
                                   allow_zero=True)
    coordinate_tolerance = _positive_float(
        coordinate_tolerance, "coordinate_tolerance", allow_zero=True)
    precision = int(precision)

    def inner(test_files, reference_files):
        try:
            pairs = _paired_file_texts(test_files, reference_files)
            maximum_energy = 0.0
            maximum_coordinate = 0.0
            maximum_location = None
            nvalues = 0
            for filename, test_text, reference_text in pairs:
                contract = _require_same_contract(
                    test_text, reference_text, filename)
                test_rows = _parse_eigenvalue_trace(
                    test_text, "test {}".format(filename))
                reference_rows = _parse_eigenvalue_trace(
                    reference_text, "reference {}".format(filename))
                _validate_eigenvalue_trajectory(
                    test_rows, contract, "test {}".format(filename))
                _validate_eigenvalue_trajectory(
                    reference_rows, contract,
                    "reference {}".format(filename))
                mismatch = _key_mismatch(test_rows, reference_rows)
                if mismatch is not None:
                    return False, "{}: eigenvalue key set mismatch; {}".format(
                        filename, mismatch)
                for key in sorted(test_rows):
                    test_coordinate, test_energy = test_rows[key]
                    reference_coordinate, reference_energy = reference_rows[key]
                    coordinate_difference = max(
                        abs(x - y) for x, y in
                        zip(test_coordinate, reference_coordinate))
                    if coordinate_difference > coordinate_tolerance:
                        return False, (
                            "{}: k-point coordinate mismatch at {}: {:.6E} "
                            "> {:.6E}"
                            .format(filename, key, coordinate_difference,
                                    coordinate_tolerance)
                        )
                    energy_difference = abs(test_energy - reference_energy) / HA2EV
                    if energy_difference > maximum_energy:
                        maximum_energy = energy_difference
                        maximum_location = (filename, key)
                    maximum_coordinate = max(
                        maximum_coordinate, coordinate_difference)
                    nvalues += 1
            message = (
                "max abs eigenvalue diff = {:.{p}E} Ha (tol = {:.{p}E} Ha), "
                "max k-point coordinate diff = {:.{p}E} over {} values"
            ).format(maximum_energy, tolerance_ha, maximum_coordinate,
                     nvalues, p=precision)
            if maximum_location is not None:
                message += ", max at {} key {}".format(*maximum_location)
            return maximum_energy <= tolerance_ha, message
        except (TypeError, ValueError) as error:
            return False, str(error)

    return inner


def iteration_summary(energy_tolerance_ev="1e-5",
                      residual_tolerance_ha="1e-8",
                      scalar_tolerance="1e-10",
                      coefficient_tolerance="1e-12", precision="3"):
    """Compare QSGW iteration summaries with field-specific tolerances."""
    energy_tolerance_ev = _positive_float(
        energy_tolerance_ev, "energy_tolerance_ev", allow_zero=True)
    residual_tolerance_ha = _positive_float(
        residual_tolerance_ha, "residual_tolerance_ha", allow_zero=True)
    scalar_tolerance = _positive_float(
        scalar_tolerance, "scalar_tolerance", allow_zero=True)
    coefficient_tolerance = _positive_float(
        coefficient_tolerance, "coefficient_tolerance", allow_zero=True)
    precision = int(precision)

    energy_fields = ("max_delta_eV", "efermi_eV", "gap_eV")
    residual_fields = ("residual_l2_Ha", "residual_max_Ha")
    scalar_fields = ("electron_count", "beta", "rcond")
    exact_fields = (
        "requested_mode", "applied_mode", "fallback", "coefficient_count",
        "converged", "fallback_reason",
    )

    def inner(test_files, reference_files):
        try:
            pairs = _paired_file_texts(test_files, reference_files)
            maxima = {name: 0.0 for name in
                      energy_fields + residual_fields + scalar_fields}
            maximum_coefficient = 0.0
            niterations = 0
            for filename, test_text, reference_text in pairs:
                _require_same_contract(test_text, reference_text, filename)
                test_rows = _parse_iteration_summary(
                    test_text, "test {}".format(filename))
                reference_rows = _parse_iteration_summary(
                    reference_text, "reference {}".format(filename))
                mismatch = _key_mismatch(test_rows, reference_rows)
                if mismatch is not None:
                    return False, "{}: iteration key set mismatch; {}".format(
                        filename, mismatch)
                for iteration in sorted(test_rows):
                    test_row = test_rows[iteration]
                    reference_row = reference_rows[iteration]
                    for field in exact_fields:
                        if test_row[field] != reference_row[field]:
                            return False, (
                                "{}: iteration {} {} mismatch: {} != {}"
                                .format(filename, iteration, field,
                                        test_row[field], reference_row[field])
                            )
                    for fields, tolerance in (
                        (energy_fields, energy_tolerance_ev),
                        (residual_fields, residual_tolerance_ha),
                        (scalar_fields, scalar_tolerance),
                    ):
                        for field in fields:
                            difference = abs(
                                test_row[field] - reference_row[field])
                            maxima[field] = max(maxima[field], difference)
                            if difference > tolerance:
                                return False, (
                                    "{}: iteration {} {} difference {:.6E} "
                                    "exceeds {:.6E}"
                                    .format(filename, iteration, field,
                                            difference, tolerance)
                                )
                    test_coefficients = test_row["coefficients"]
                    reference_coefficients = reference_row["coefficients"]
                    if len(test_coefficients) != len(reference_coefficients):
                        return False, (
                            "{}: iteration {} coefficient count mismatch"
                            .format(filename, iteration)
                        )
                    for index, (test_value, reference_value) in enumerate(
                            zip(test_coefficients, reference_coefficients)):
                        difference = abs(test_value - reference_value)
                        maximum_coefficient = max(
                            maximum_coefficient, difference)
                        if difference > coefficient_tolerance:
                            return False, (
                                "{}: iteration {} coefficient {} difference "
                                "{:.6E} exceeds {:.6E}"
                                .format(filename, iteration, index, difference,
                                        coefficient_tolerance)
                            )
                    coefficient_l1_difference = abs(
                        test_row["coefficient_l1"] -
                        reference_row["coefficient_l1"])
                    if coefficient_l1_difference > coefficient_tolerance:
                        return False, (
                            "{}: iteration {} coefficient_l1 difference "
                            "{:.6E} exceeds {:.6E}"
                            .format(filename, iteration,
                                    coefficient_l1_difference,
                                    coefficient_tolerance)
                        )
                    niterations += 1

            message = (
                "max gap diff = {:.{p}E} eV, max eigenvalue-change diff = "
                "{:.{p}E} eV, max residual diff = {:.{p}E} Ha, max "
                "coefficient diff = {:.{p}E} over {} iterations"
            ).format(
                maxima["gap_eV"], maxima["max_delta_eV"],
                max(maxima[name] for name in residual_fields),
                maximum_coefficient, niterations, p=precision)
            return True, message
        except (TypeError, ValueError) as error:
            return False, str(error)

    return inner


def _positive_float(value, label, allow_zero=False):
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (
            result == 0.0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError("{} must be finite and {}".format(label, relation))
    return result


def _as_bool(value, label):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError("{} must be a boolean".format(label))


def _paired_file_texts(test_files, reference_files):
    filenames = set(test_files) | set(reference_files)
    if not filenames:
        raise ValueError("no files found")
    pairs = []
    for filename in sorted(filenames, key=str):
        if filename not in test_files or filename not in reference_files:
            raise ValueError("missing file {}".format(filename))
        pairs.append((filename,
                      _single_text(test_files[filename], filename),
                      _single_text(reference_files[filename], filename)))
    return pairs


def _parse_band_filename(filename):
    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    match = re.fullmatch(
        r"QSGW_band_spin_([1-9][0-9]*)_([1-9][0-9]*)\.dat",
        basename)
    if match is None:
        raise ValueError("invalid QSGW band filename: {}".format(filename))
    return int(match.group(1)), int(match.group(2))


def _parse_band_table(text, label):
    rows = []
    n_bands = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 6 or (len(fields) - 4) % 2 != 0:
            raise ValueError(
                "{}:{}: invalid QSGW band column count"
                .format(label, line_number))
        try:
            index = int(fields[0])
        except ValueError as error:
            raise ValueError(
                "{}:{}: invalid k-point index"
                .format(label, line_number)) from error
        if index != len(rows) + 1:
            raise ValueError(
                "{}:{}: non-contiguous k-point index"
                .format(label, line_number))
        row_n_bands = (len(fields) - 4) // 2
        if n_bands is None:
            n_bands = row_n_bands
        elif row_n_bands != n_bands:
            raise ValueError("{}: inconsistent band counts".format(label))
        coordinate = tuple(_finite_float(value) for value in fields[1:4])
        occupations = tuple(
            _finite_float(fields[4 + 2 * band])
            for band in range(row_n_bands))
        energies = tuple(
            _finite_float(fields[5 + 2 * band])
            for band in range(row_n_bands))
        if any(value < 0.0 for value in occupations):
            raise ValueError(
                "{}:{}: negative occupation".format(label, line_number))
        rows.append((coordinate, energies))
    if not rows:
        raise ValueError("{}: empty QSGW band table".format(label))
    return tuple(rows)


def _band_gap(rows, occupied_bands):
    valence_maximum = max(
        energies[occupied_bands - 1] for _, energies in rows)
    conduction_minimum = min(
        energies[occupied_bands] for _, energies in rows)
    return conduction_minimum - valence_maximum


def _validate_band_trajectories(trajectories):
    spins = sorted(trajectories)
    if spins != list(range(1, spins[-1] + 1)):
        raise ValueError(
            "QSGW band spin indices are not continuous: {}".format(spins))
    expected = None
    for spin in spins:
        iterations = sorted(trajectories[spin])
        continuous = list(range(1, iterations[-1] + 1))
        if iterations != continuous:
            raise ValueError(
                "QSGW band iterations are not continuous for spin {}: {}"
                .format(spin, iterations))
        if expected is None:
            expected = iterations
        elif iterations != expected:
            raise ValueError(
                "QSGW band iteration sets differ between spins")
    return len(expected)


def _single_text(raw, filename):
    if isinstance(raw, str):
        return raw
    values = list(raw)
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("{} must contain one complete trace".format(filename))
    return values[0]


def _require_header(text, expected, label):
    comments = [line.strip() for line in text.splitlines()
                if line.strip().startswith("#")]
    if expected not in comments:
        raise ValueError("{}: missing trace header".format(label))


def _require_same_contract(test_text, reference_text, filename):
    test_contract = _parse_contract(
        test_text, "test {}".format(filename))
    reference_contract = _parse_contract(
        reference_text, "reference {}".format(filename))
    if test_contract != reference_contract:
        differing = sorted(
            key for key in set(test_contract) | set(reference_contract)
            if test_contract.get(key) != reference_contract.get(key)
        )
        raise ValueError("{}: QSGW contract differs for {}".format(
            filename, differing))
    return test_contract


def _parse_contract(text, label):
    values = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split(None, 1)
        if len(fields) != 2 or fields[0] not in CONTRACT_KEYS:
            continue
        key, value = fields[0], fields[1].strip()
        if key in values:
            raise ValueError("{}:{}: duplicate QSGW contract key {}".format(
                label, line_number, key))
        values[key] = value

    missing = sorted(REQUIRED_CONTRACT_KEYS - set(values))
    if missing:
        raise ValueError("{}: missing QSGW contract keys {}".format(
            label, missing))
    try:
        version = int(values["qsgw_contract_version"])
    except ValueError as error:
        raise ValueError("{}: invalid QSGW contract version".format(
            label)) from error
    if version not in (5, 6):
        raise ValueError("{}: unsupported QSGW contract version {}".format(
            label, version))
    values["qsgw_contract_version"] = version

    if version == 5:
        if "headwing" not in values:
            raise ValueError(
                "{}: missing QSGW contract keys ['headwing']".format(label))
        if "head" in values or "wing" in values:
            raise ValueError(
                "{}: contract-v5 has split head/wing fields".format(label))
    else:
        if "headwing" in values:
            raise ValueError(
                "{}: contract-v6 has legacy headwing field; "
                "use split head/wing fields".format(label))
        missing_headwing = [
            key for key in ("head", "wing") if key not in values
        ]
        if missing_headwing:
            raise ValueError(
                "{}: missing QSGW contract keys {}".format(
                    label, missing_headwing))
        if values["wing"] != "disabled_stage1":
            raise ValueError(
                "{}: contract-v6 enables unsupported iterative wing".format(
                    label))
        values["headwing"] = values["head"]

    expected = {
        "fixed_basis": "immutable_mf0",
        "live_update": "eigenvalues_wfc",
    }
    for key, required in expected.items():
        if values[key] != required:
            raise ValueError("{}: invalid {} QSGW contract".format(
                label, key))

    symmetry = values["symmetry"]
    if symmetry != "unsupported_full_bz_only" and re.fullmatch(
            r"exx_(?:on|off)_gw_(?:on|off)_rpa_(?:on|off)",
            symmetry) is None:
        raise ValueError("{}: invalid symmetry QSGW contract".format(label))

    headwing_velocity = {
        "disabled_stage1": "disabled_stage1",
        "scf_grid_analytic_live": "fixed_basis_rotation",
        "independent_full_grid_analytic_live": "live_ao_fourier_rotation",
    }
    headwing = values["headwing"]
    if headwing not in headwing_velocity or \
            values["velocity"] != headwing_velocity[headwing]:
        raise ValueError("{}: inconsistent headwing/velocity QSGW contract".format(
            label))

    hartree = values["hartree"]
    if hartree == "disabled_stage1":
        if "hartree_coulomb" in values or "hartree_normalization" in values:
            raise ValueError("{}: disabled Hartree has extra contract fields".format(
                label))
    elif hartree == "delta_density":
        missing_hartree = [key for key in (
            "hartree_coulomb", "hartree_normalization") if key not in values]
        if missing_hartree:
            raise ValueError("{}: enabled Hartree is missing contract fields {}".format(
                label, missing_hartree))
        if values["hartree_coulomb"] not in ("full", "truncated"):
            raise ValueError("{}: invalid Hartree Coulomb contract".format(label))
        if values["hartree_normalization"] not in (
                "weighted_occupations", "legacy_extra_inverse_nk"):
            raise ValueError("{}: invalid Hartree normalization contract".format(
                label))
    else:
        raise ValueError("{}: invalid Hartree QSGW contract".format(label))

    if values["band"] not in (
            "disabled_stage1", "fixed_reference_rotation_live"):
        raise ValueError("{}: invalid band QSGW contract".format(label))
    cut_keys = (
        "qsgw_band0_unoccupied_keep",
        "qsgw_band0_cut_mode",
        "qsgw_band0_cut_shift_ha",
    )
    if version == 6:
        if "h_qsgw_cut" not in values:
            raise ValueError("{}: missing QSGW contract keys ['h_qsgw_cut']".format(
                label))
        cut_contract = values["h_qsgw_cut"]
        if cut_contract == "disabled_non_band":
            if values["band"] != "disabled_stage1" or any(
                    key in values for key in cut_keys):
                raise ValueError("{}: inconsistent disabled H_QSGW cut contract".format(
                    label))
        elif cut_contract == "band_postprocess":
            if values["band"] != "fixed_reference_rotation_live":
                raise ValueError("{}: H_QSGW cut requires band postprocessing".format(
                    label))
            missing_cut = [key for key in cut_keys if key not in values]
            if missing_cut:
                raise ValueError("{}: enabled H_QSGW cut is missing contract fields {}".format(
                    label, missing_cut))
            try:
                unoccupied_keep = int(values["qsgw_band0_unoccupied_keep"])
                cut_mode = int(values["qsgw_band0_cut_mode"])
            except ValueError as error:
                raise ValueError("{}: invalid H_QSGW cut integer contract".format(
                    label)) from error
            cut_shift = _finite_float(values["qsgw_band0_cut_shift_ha"])
            if unoccupied_keep < 0 or cut_mode not in (0, 1, 2):
                raise ValueError("{}: invalid H_QSGW cut contract".format(label))
            values["qsgw_band0_unoccupied_keep"] = unoccupied_keep
            values["qsgw_band0_cut_mode"] = cut_mode
            values["qsgw_band0_cut_shift_ha"] = cut_shift
        else:
            raise ValueError("{}: invalid H_QSGW cut contract".format(label))
    elif "h_qsgw_cut" in values or any(key in values for key in cut_keys):
        raise ValueError("{}: QSGW contract version 5 has H_QSGW cut fields".format(
            label))
    if not values["qsgw_input_contract"]:
        raise ValueError("{}: empty QSGW input contract path".format(label))
    sha256 = values["qsgw_input_contract_sha256"].lower()
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("{}: invalid QSGW input contract SHA256".format(label))
    values["qsgw_input_contract_sha256"] = sha256
    if values["qsgw_mixer"] not in ("none", "linear"):
        raise ValueError("{}: invalid QSGW mixer contract".format(label))
    beta = _finite_float(values["qsgw_mixing_beta"])
    if not 0.0 < beta <= 1.0:
        raise ValueError("{}: invalid QSGW mixing beta".format(label))
    values["qsgw_mixing_beta"] = beta
    return values


def _parse_matrix_trace(text, label):
    _require_header(text, MATRIX_HEADER, label)
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 11:
            raise ValueError("{}:{}: matrix row has {} columns".format(
                label, line_number, len(fields)))
        try:
            iteration = int(fields[0])
            channel = int(fields[1])
            component = fields[2]
            spin = int(fields[3])
            kpoint = int(fields[4])
            frequency_index = int(fields[5])
            frequency = _finite_float(fields[6])
            row = int(fields[7])
            column = int(fields[8])
            value = complex(_finite_float(fields[9]),
                            _finite_float(fields[10]))
        except ValueError as error:
            raise ValueError("{}:{}: invalid matrix row".format(
                label, line_number)) from error
        if iteration < 0 or channel not in (0, 1, 2) or min(
                spin, kpoint, row, column) < 0 or frequency_index < -1:
            raise ValueError("{}:{}: invalid matrix index".format(
                label, line_number))
        if not component or frequency < 0.0 or (
                frequency_index == -1 and frequency != 0.0):
            raise ValueError("{}:{}: invalid matrix metadata".format(
                label, line_number))
        key = (iteration, channel, component, spin, kpoint,
               frequency_index, row, column)
        if key in rows:
            raise ValueError("{}:{}: duplicate matrix row {}".format(
                label, line_number, key))
        rows[key] = (frequency, value)
    if not rows:
        raise ValueError("{}: no matrix rows found".format(label))

    grouped = {}
    for key, item in rows.items():
        block_key = key[:6]
        row_column = key[6:]
        grouped.setdefault(block_key, {})[row_column] = item

    blocks = {}
    for block_key, entries in grouped.items():
        nrows = max(index[0] for index in entries) + 1
        ncolumns = max(index[1] for index in entries) + 1
        expected = {(row, column) for row in range(nrows)
                    for column in range(ncolumns)}
        if set(entries) != expected:
            raise ValueError("{}: incomplete matrix block {}".format(
                label, block_key))
        frequencies = {item[0] for item in entries.values()}
        if len(frequencies) != 1:
            raise ValueError("{}: inconsistent frequency in block {}".format(
                label, block_key))
        matrix = [[entries[(row, column)][1]
                   for column in range(ncolumns)]
                  for row in range(nrows)]
        blocks[block_key] = (next(iter(frequencies)), matrix)
    return blocks


def _validate_matrix_trajectory(blocks, contract, label):
    iterations = _continuous_iterations(
        {key[0] for key in blocks}, label)
    headwing = contract["headwing"]
    hartree = contract["hartree"] == "delta_density"
    band = contract["band"] == "fixed_reference_rotation_live"
    independent_headwing = headwing == "independent_full_grid_analytic_live"

    expected_channels = {0}
    if band:
        expected_channels.add(1)
    if independent_headwing:
        expected_channels.add(2)
    observed_channels = {key[1] for key in blocks}
    if observed_channels != expected_channels:
        raise ValueError(
            "{}: matrix channel set differs from the QSGW contract: {} != {}"
            .format(label, sorted(observed_channels),
                    sorted(expected_channels)))

    frequency_components = {"sigma_c_iw", "head_tensor"}
    components = {}
    for key in blocks:
        iteration, channel, component = key[:3]
        frequency_index = key[5]
        components.setdefault((iteration, channel), set()).add(component)
        if component in frequency_components:
            if frequency_index < 0:
                raise ValueError(
                    "{}: frequency component {} has a static index"
                    .format(label, component))
        elif frequency_index != -1:
            raise ValueError(
                "{}: static component {} has a frequency index"
                .format(label, component))

    initial_wavefunctions = {}
    for channel in sorted(expected_channels):
        initial = components.get((0, channel), set())
        wavefunctions = _wavefunction_components(initial, label, 0, channel)
        initial_wavefunctions[channel] = wavefunctions

        if channel in (0, 1):
            required = {"h0", "vxc_dft", "occupation"}
        else:
            required = {
                "h0", "occupation", "head_tensor",
                "velocity_x", "velocity_y", "velocity_z",
            }
        missing = required - initial
        if missing:
            raise ValueError(
                "{}: iteration 0 channel {} is missing required components {}"
                .format(label, channel, sorted(missing)))

    fourier_diagnostics = {
        "basis_inverse_residual",
        "basis_condition_estimate",
        "fourier_orthogonality_residual",
        "source_roundtrip_relative_error",
        "target_hermiticity_error",
        "target_relative_hermiticity_error",
        "repaired_target_hermiticity_error",
    }
    for iteration in iterations[1:]:
        for channel in sorted(expected_channels):
            present = components.get((iteration, channel), set())
            wavefunctions = _wavefunction_components(
                present, label, iteration, channel)
            if wavefunctions != initial_wavefunctions[channel]:
                raise ValueError(
                    "{}: iteration {} channel {} wavefunction components "
                    "differ from iteration zero"
                    .format(label, iteration, channel))

            if channel == 0:
                required = {
                    "sigma_c_iw", "exx", "vc", "raw_h", "mixed_h",
                    "rotation_u", "occupation",
                }
                if hartree:
                    required.add("delta_vh")
            elif channel == 1:
                required = {
                    "exx", "vc", "raw_h", "mixed_h", "rotation_u",
                    "occupation",
                }
                if hartree:
                    required.add("delta_vh")
            else:
                required = {
                    "head_tensor", "projected_h", "rotation_u",
                    "occupation", "velocity_x", "velocity_y", "velocity_z",
                } | fourier_diagnostics
            missing = required - present
            if missing:
                raise ValueError(
                    "{}: iteration {} channel {} is missing required "
                    "components {}"
                    .format(label, iteration, channel, sorted(missing)))


def _wavefunction_components(components, label, iteration, channel):
    wavefunctions = {
        component for component in components
        if component.startswith("wfc_spinor")
    }
    if not wavefunctions:
        raise ValueError(
            "{}: iteration {} channel {} has no wavefunction components"
            .format(label, iteration, channel))
    try:
        indices = sorted(int(component[len("wfc_spinor"):])
                         for component in wavefunctions)
    except ValueError as error:
        raise ValueError(
            "{}: invalid wavefunction component label".format(label)) from error
    if indices != list(range(len(indices))):
        raise ValueError(
            "{}: wavefunction spinor components are not contiguous".format(
                label))
    return wavefunctions


def _parse_eigenvalue_trace(text, label):
    _require_header(text, EIGENVALUE_HEADER, label)
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError("{}:{}: eigenvalue row has {} columns".format(
                label, line_number, len(fields)))
        try:
            iteration, channel, spin, kpoint = map(int, fields[:4])
            coordinate = tuple(_finite_float(value) for value in fields[4:7])
            band = int(fields[7])
            energy = _finite_float(fields[8])
        except ValueError as error:
            raise ValueError("{}:{}: invalid eigenvalue row".format(
                label, line_number)) from error
        if min(iteration, spin, kpoint, band) < 0 or channel not in (0, 1, 2):
            raise ValueError("{}:{}: invalid eigenvalue index".format(
                label, line_number))
        key = (iteration, channel, spin, kpoint, band)
        if key in rows:
            raise ValueError("{}:{}: duplicate eigenvalue row {}".format(
                label, line_number, key))
        rows[key] = (coordinate, energy)
    if not rows:
        raise ValueError("{}: no eigenvalue rows found".format(label))
    return rows


def _validate_eigenvalue_trajectory(rows, contract, label):
    iterations = _continuous_iterations({key[0] for key in rows}, label)
    expected_channels = {0}
    if contract["band"] == "fixed_reference_rotation_live":
        expected_channels.add(1)
    if contract["headwing"] == "independent_full_grid_analytic_live":
        expected_channels.add(2)
    observed_channels = {key[1] for key in rows}
    if observed_channels != expected_channels:
        raise ValueError(
            "{}: eigenvalue channel set differs from the QSGW contract"
            .format(label))

    baseline = {}
    for channel in sorted(expected_channels):
        keys = {
            key[1:] for key in rows
            if key[0] == 0 and key[1] == channel
        }
        if not keys:
            raise ValueError(
                "{}: iteration zero channel {} has no eigenvalues".format(
                    label, channel))
        baseline[channel] = keys

    for iteration in iterations:
        for channel in sorted(expected_channels):
            keys = {
                key[1:] for key in rows
                if key[0] == iteration and key[1] == channel
            }
            if keys != baseline[channel]:
                raise ValueError(
                    "{}: iteration {} channel {} eigenvalue layout differs "
                    "from iteration zero"
                    .format(label, iteration, channel))


def _continuous_iterations(iterations, label):
    observed = sorted(iterations)
    if not observed or observed[0] != 0:
        raise ValueError("{}: iteration zero is missing".format(label))
    if observed[-1] < 1:
        raise ValueError("{}: no completed QSGW iteration is present".format(
            label))
    expected = list(range(observed[-1] + 1))
    if observed != expected:
        raise ValueError("{}: QSGW iterations are not continuous: {}".format(
            label, observed))
    return observed


def _parse_iteration_summary(text, label):
    _require_header(text, SUMMARY_HEADER, label)
    rows = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 17:
            raise ValueError("{}:{}: summary row has {} columns".format(
                label, line_number, len(fields)))
        try:
            iteration = int(fields[0])
            row = {
                "max_delta_eV": _finite_float(fields[1]),
                "residual_l2_Ha": _finite_float(fields[2]),
                "residual_max_Ha": _finite_float(fields[3]),
                "efermi_eV": _finite_float(fields[4]),
                "gap_eV": _finite_float(fields[5]),
                "electron_count": _finite_float(fields[6]),
                "requested_mode": int(fields[7]),
                "applied_mode": int(fields[8]),
                "beta": _finite_float(fields[9]),
                "fallback": int(fields[10]),
                "rcond": _finite_float(fields[11]),
                "coefficient_l1": _finite_float(fields[12]),
                "coefficient_count": int(fields[13]),
                "converged": int(fields[14]),
                "coefficients": _parse_coefficients(fields[15]),
                "fallback_reason": fields[16],
            }
        except ValueError as error:
            raise ValueError("{}:{}: invalid summary row".format(
                label, line_number)) from error
        if iteration < 0 or any(row[name] < 0.0 for name in (
                "max_delta_eV", "residual_l2_Ha", "residual_max_Ha",
                "gap_eV", "electron_count", "beta", "rcond",
                "coefficient_l1")):
            raise ValueError("{}:{}: invalid summary value".format(
                label, line_number))
        if row["coefficient_count"] != len(row["coefficients"]):
            raise ValueError("{}:{}: coefficient count is inconsistent".format(
                label, line_number))
        coefficient_l1 = sum(abs(value) for value in row["coefficients"])
        if abs(coefficient_l1 - row["coefficient_l1"]) > 1.0e-12 * max(
                1.0, coefficient_l1):
            raise ValueError("{}:{}: coefficient L1 norm is inconsistent".format(
                label, line_number))
        if iteration in rows:
            raise ValueError("{}:{}: duplicate iteration {}".format(
                label, line_number, iteration))
        rows[iteration] = row
    if not rows:
        raise ValueError("{}: no iteration rows found".format(label))
    iterations = sorted(rows)
    if iterations[0] != 0:
        raise ValueError("{}: iteration zero is missing".format(label))
    if iterations[-1] < 1:
        raise ValueError("{}: no completed QSGW iteration is present".format(
            label))
    expected = list(range(iterations[-1] + 1))
    if iterations != expected:
        raise ValueError("{}: QSGW iterations are not continuous: {}".format(
            label, iterations))
    return rows


def _parse_coefficients(value):
    if value == "none":
        return tuple()
    coefficients = tuple(_finite_float(item) for item in value.split(","))
    if not coefficients:
        raise ValueError("empty coefficient list")
    return coefficients


def _finite_float(value):
    result = float(value.replace("D", "E").replace("d", "E"))
    if not math.isfinite(result):
        raise ValueError("non-finite value")
    return result


def _key_mismatch(test, reference):
    test_keys = set(test)
    reference_keys = set(reference)
    if test_keys == reference_keys:
        return None
    missing = sorted(reference_keys - test_keys, key=str)[:3]
    extra = sorted(test_keys - reference_keys, key=str)[:3]
    return "missing={}, extra={}".format(missing, extra)


def _shape(matrix):
    return len(matrix), len(matrix[0]) if matrix else 0


def _frobenius(matrix):
    return math.sqrt(sum(abs(value) ** 2
                         for row in matrix for value in row))


def _difference_frobenius(test, reference):
    return math.sqrt(sum(
        abs(test[row][column] - reference[row][column]) ** 2
        for row in range(len(test))
        for column in range(len(test[row]))
    ))


def _hermiticity_residual(matrix):
    nrows, ncolumns = _shape(matrix)
    if nrows != ncolumns:
        raise ValueError("Hermiticity check requires a square matrix")
    return max(
        abs(matrix[row][column] - matrix[column][row].conjugate())
        for row in range(nrows) for column in range(ncolumns)
    )


def _unitarity_residual(matrix):
    nrows, ncolumns = _shape(matrix)
    if nrows != ncolumns:
        raise ValueError("unitarity check requires a square matrix")
    residual = 0.0
    for row in range(ncolumns):
        for column in range(ncolumns):
            value = sum(matrix[index][row].conjugate() *
                        matrix[index][column] for index in range(nrows))
            if row == column:
                value -= 1.0
            residual = max(residual, abs(value))
    return residual
