param(
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Manifest = (Join-Path $Repository 'qsgw-rebase-manifest.json'),
    [string]$TemplatePath = 'F:\AI_Workspace\Theoretical-Physics\.agents\skills\workflow\librpa-qsgw-upstream-rebase\templates\qsgw-rebase-manifest.json'
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)

function Convert-TemplateValue {
    param($Value)

    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [string]) {
        if ($Value -match '(?i)(replace_with_|replace-with-)') {
            return $null
        }
        return $Value
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $result = [ordered]@{}
        foreach ($property in $Value.PSObject.Properties) {
            $result[$property.Name] = Convert-TemplateValue $property.Value
        }
        return $result
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $Value.Keys) {
            $result[$key] = Convert-TemplateValue $Value[$key]
        }
        return $result
    }
    if ($Value -is [System.Collections.IEnumerable] -and
        $Value -isnot [string]) {
        $items = @($Value | ForEach-Object { Convert-TemplateValue $_ })
        return ,$items
    }
    return $Value
}

function Get-Sha256 {
    param([string]$RelativePath)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath (
        Join-Path $Repository $RelativePath)).Hash.ToLowerInvariant()
}

function New-Contract {
    param(
        [string]$Basis,
        [string]$Shape,
        [string]$Units,
        [string]$Ownership,
        [string]$Normalization
    )
    return [ordered]@{
        basis = $Basis
        shape = $Shape
        units = $Units
        ownership = $Ownership
        normalization = $Normalization
    }
}

$templateObject = Get-Content -Raw -Encoding UTF8 -LiteralPath $TemplatePath |
    ConvertFrom-Json
$data = Convert-TemplateValue $templateObject
while ($data -is [array] -and $data.Count -eq 1) {
    $data = $data[0]
}
if ($data -isnot [System.Collections.IDictionary]) {
    throw "Template conversion did not produce a JSON object: $($data.GetType().FullName)"
}
$inventoryPath = 'qsgw-rebase-evidence/git/upstream-hunk-inventory.json'
$inventory = Get-Content -Raw -Encoding UTF8 -LiteralPath (
    Join-Path $Repository $inventoryPath) | ConvertFrom-Json

$upstreamOld = 'b484f2a9a252c8a7169c67e9781da6f9c07c310a'
$upstreamNew = '1376ee4f45a7611a55c5b92c4ba41409d515bcea'
$qsgwOld = 'cb2940201b1f44b1c38b90169048b7d060e44732'
$candidateSource = '7d69a18cff419be3139d61b8405a8dbca6b53fd8'

$data['repository'] = [ordered]@{
    path = $Repository.Replace('\', '/')
    worktree = $Repository.Replace('\', '/')
    branch = 'codex/qsgw-independent-upstream-1376-20260714'
    status_snapshot = 'qsgw-rebase-evidence/git/status.txt'
    remotes_snapshot = 'qsgw-rebase-evidence/git/remotes.txt'
    commits = [ordered]@{
        branch_head = [ordered]@{
            hash = $candidateSource
            reference = 'qsgw-rebase-evidence/git/candidate-source.txt'
        }
        upstream_old = [ordered]@{
            hash = $upstreamOld
            reference = 'qsgw-rebase-evidence/git/upstream-old.txt'
        }
        upstream_new = [ordered]@{
            hash = $upstreamNew
            reference = 'qsgw-rebase-evidence/git/upstream-new.txt'
        }
        qsgw_old = [ordered]@{
            hash = $qsgwOld
            reference = 'qsgw-rebase-evidence/git/qsgw-old.txt'
        }
        rebase_head = [ordered]@{
            hash = $candidateSource
            reference = 'qsgw-rebase-evidence/git/candidate-source.txt'
        }
    }
}

$comparatorPath = 'regression_tests/backend/comparisons/cmp_qsgw.py'
$environmentPath = 'qsgw-rebase-evidence/environment/local-freeze.json'
$data['provenance'] = [ordered]@{
    source = [ordered]@{
        hash = $candidateSource
        reference = 'qsgw-rebase-evidence/git/candidate-source.txt'
    }
    executable = [ordered]@{
        path = $null
        sha256 = $null
        source_commit = $candidateSource
        reference = 'pending: clean remote Gate-0 build has not been created'
    }
    dataset = [ordered]@{
        id = $null
        path = $null
        sha256 = $null
        reference = 'pending: immutable Si-k444 no-symmetry dataset manifest'
    }
    inputs = @(
        [ordered]@{
            id = 'librpa-input'
            path = $null
            sha256 = $null
            reference = 'pending: clean Si-k444 no-symmetry librpa.in'
        }
    )
    environment = [ordered]@{
        host = 'pending remote fish/dongfang compute host'
        os = 'pending remote Linux capture'
        compiler = [ordered]@{
            name = $null
            version = $null
            reference = 'pending: Gate-0 compiler capture'
        }
        libraries = @(
            [ordered]@{
                name = 'LibRI'
                version = $null
                reference = 'pending: exact LibRI commit and dirty-diff capture'
            }
        )
        mpi = [ordered]@{
            implementation = $null
            version = $null
            ranks = 1
        }
        omp_threads = 1
        variables = [ordered]@{
            OMP_PROC_BIND = 'close'
            OMP_PLACES = 'cores'
            LIBRI_DETERMINISTIC_REDUCTION = '1'
        }
        random_seed = $null
        capture = [ordered]@{
            path = $environmentPath
            sha256 = Get-Sha256 $environmentPath
            reference = 'local freeze only; remote environment remains pending'
        }
    }
    parameters = [ordered]@{
        mix_mode = 'disabled'
        head = $false
        wing = $false
        hartree = $false
        band = $false
        use_symmetry_gw = $false
        use_symmetry_exx = $false
        max_iter = 1
    }
    comparator = [ordered]@{
        name = 'cmp_qsgw trace comparator'
        version = 'contract-v5 candidate 7d69a18c'
        sha256 = Get-Sha256 $comparatorPath
        reference = $comparatorPath
        tolerances = [ordered]@{
            eigenvalue_Ha = 1e-6
            gap_eV = 1e-5
            matrix_relative_frobenius = 1e-8
            hermiticity = 1e-10
            orthogonality = 1e-10
        }
    }
}

function Set-ComparisonSideCommit {
    param(
        [System.Collections.IDictionary]$Side,
        [string]$Commit,
        [string]$Reference
    )

    $Side['source']['hash'] = $Commit
    $Side['source']['reference'] = $Reference
    $Side['executable']['source_commit'] = $Commit
}

foreach ($benchmark in @($data['benchmarks'])) {
    switch ($benchmark['id']) {
        'g0w0-upstream-vs-rebased' {
            Set-ComparisonSideCommit $benchmark['definition']['side_a'] $upstreamNew 'qsgw-rebase-evidence/git/upstream-new.txt'
            Set-ComparisonSideCommit $benchmark['definition']['side_b'] $candidateSource 'qsgw-rebase-evidence/git/candidate-source.txt'
        }
        'qsgw-iter0-vs-upstream' {
            Set-ComparisonSideCommit $benchmark['definition']['side_a'] $upstreamNew 'qsgw-rebase-evidence/git/upstream-new.txt'
            Set-ComparisonSideCommit $benchmark['definition']['side_b'] $candidateSource 'qsgw-rebase-evidence/git/candidate-source.txt'
        }
        'solid-qsgw-no-mixing' {
            Set-ComparisonSideCommit $benchmark['definition']['side_a'] $qsgwOld 'qsgw-rebase-evidence/git/qsgw-old.txt'
            Set-ComparisonSideCommit $benchmark['definition']['side_b'] $candidateSource 'qsgw-rebase-evidence/git/candidate-source.txt'
        }
    }
}

foreach ($pair in @($data['control_pairs'])) {
    Set-ComparisonSideCommit $pair['a'] $candidateSource 'qsgw-rebase-evidence/git/candidate-source.txt'
    Set-ComparisonSideCommit $pair['b'] $candidateSource 'qsgw-rebase-evidence/git/candidate-source.txt'
}

$commitHashes = @($inventory.commits.commit)
$commitChangeMap = @(
    $inventory.commits | ForEach-Object {
        [ordered]@{
            commit = $_.commit
            change_ids = @($_.change_ids)
        }
    }
)
$classifiedIds = @($inventory.changes.id)
$data['upstream_inventory'] = [ordered]@{
    old_commit = $upstreamOld
    new_commit = $upstreamNew
    commit_list = [ordered]@{
        path = 'qsgw-rebase-evidence/git/upstream-commit-list.txt'
        sha256 = Get-Sha256 'qsgw-rebase-evidence/git/upstream-commit-list.txt'
        reference = 'git rev-list --reverse upstream_old..upstream_new'
    }
    name_status = [ordered]@{
        path = 'qsgw-rebase-evidence/git/upstream-name-status.txt'
        sha256 = Get-Sha256 'qsgw-rebase-evidence/git/upstream-name-status.txt'
        reference = 'git diff --name-status upstream_old..upstream_new'
    }
    hunk_inventory = [ordered]@{
        path = $inventoryPath
        sha256 = Get-Sha256 $inventoryPath
        reference = 'reviewed semantic hunk inventory with exact commit/change coverage'
    }
    commit_hashes = $commitHashes
    commit_change_map = $commitChangeMap
    classified_change_ids = $classifiedIds
    coverage_assertion = 'complete'
}

$data['protected_shared_paths'] = @(
    'src/',
    'src/api/',
    'src/core/',
    'src/gw/',
    'src/app_gw/',
    'src/exx/',
    'src/chi0/',
    'src/epsilon/',
    'src/libri/',
    'src/symmetry/',
    'src/matrix_m/',
    'src/parallel_mpi/',
    'src/distributed_matrix/'
)

$observerIds = [ordered]@{
    U0 = @('build-upstream-regressions')
    U1 = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
    U2 = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
    U3 = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
}
$effectDomains = [ordered]@{
    U0 = 'none'
    U1 = 'upstream_g0w0_inherited'
    U2 = 'qsgw_adapter'
    U3 = 'shared_conflict'
}
$dispositions = [ordered]@{
    U0 = 'accept_upstream'
    U1 = 'accept_upstream'
    U2 = 'adapt_qsgw_only'
    U3 = 'approved_shared_interface'
}
$semanticFields = @(
    'api', 'matrix_layout', 'basis', 'units', 'normalization',
    'mpi_ownership', 'k_point_weights', 'occupations',
    'call_ordering', 'symmetry'
)

$data['upstream_changes'] = @(
    foreach ($change in $inventory.changes) {
        $semantic = [ordered]@{}
        foreach ($field in $semanticFields) {
            if ($change.semantic_impact -is [System.Management.Automation.PSCustomObject]) {
                $semantic[$field] = $change.semantic_impact.$field
            }
            elseif ($change.classification -eq 'U0') {
                $semantic[$field] = 'not_applicable'
            }
            else {
                $semantic[$field] = 'unchanged'
            }
        }
        $semantic.evidence = $inventoryPath
        $adapterObservers = @()
        if ($change.classification -eq 'U2') {
            $adapterObservers = @('qsgw-iter0-vs-upstream')
        }
        $record = [ordered]@{
            id = $change.id
            commit = $change.commit
            commit_reference = $inventoryPath
            classification = $change.classification
            effect_domain = $effectDomains[$change.classification]
            changed_paths = @($change.paths)
            summary = $change.qsgw_reachability
            disposition = $dispositions[$change.classification]
            semantic_audit = $semantic
            benchmark_observer_ids = @($observerIds[$change.classification])
            qsgw_adapter_observer_ids = $adapterObservers
            approval_ids = @()
        }
        if ($change.classification -eq 'U3') {
            $record.approval_ids = @('APPROVAL-U3-HEAD-MATRIX-GETTER-001')
            $record.u3_impact_report = [ordered]@{
                formula_mapping = 'qsgw-rebase-evidence/impact/formula-to-code.md#f-head-tensor-read'
                call_chain = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md#call-chain'
                candidate_diff = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.patch'
                affected_scope = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md#numerical-and-abi-impact'
                benchmark_design = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md#required-observers'
            }
        }
        $record
    }
)

$sameContract = New-Contract 'upstream fixed basis' 'upstream-defined' 'Hartree' 'Dataset communicator' 'upstream-defined'
$data['formula_code_impacts'] = @(
    [ordered]@{
        formula_id = 'F-BVK-REMAP-LOG'
        formula = 'The BvK remap is invariant under verbosity changes.'
        upstream_symbols = @('api::build_band_bvk_remap')
        qsgw_symbols = @('run_qsgw')
        upstream_change_ids = @('UP-BVK-LOG-001')
        upstream_commits = @('bb0e27624bc5c4dcb923a2ddd353d6e087d0c0e2')
        contract_before = $sameContract
        contract_after = $sameContract
        classification = 'U1'
        required_change = 'none'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-HEAD-GAMMA-VOLUME'
        formula = 'Omega_Gamma = Omega_reciprocal / N_BvK.'
        upstream_symbols = @('rpa_headwing_gamma_cell_volume', 'diele_func::cal_eps')
        qsgw_symbols = @('refresh_headwing')
        upstream_change_ids = @('UP-HEAD-GAMMA-VOLUME-001')
        upstream_commits = @('72559e92cf8ad54fad70466598657ebcd5cf265a')
        contract_before = New-Contract 'head Cartesian tensor' '3x3 per frequency' 'Hartree-derived dielectric units' 'shared dielectric object' 'active k-count'
        contract_after = New-Contract 'head Cartesian tensor' '3x3 per frequency' 'Hartree-derived dielectric units' 'shared dielectric object' 'complete BvK-cell count'
        classification = 'U1'
        required_change = 'none'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'head-only')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-NOSYM-SYMMETRY-BYPASS'
        formula = 'Full-grid inputs with all symmetry switches false bypass crystal k-star restoration.'
        upstream_symbols = @('find_symmetry_atom_target', 'read_headwing_input', 'diele_func::cal_head', 'diele_func::cal_wing')
        qsgw_symbols = @('validate_stage_one_contract', 'refresh_headwing')
        upstream_change_ids = @('UP-SYMMETRY-TEXT-TOL-001', 'UP-HEADWING-SYMMETRY-ROUTE-001')
        upstream_commits = @('a209b9e5abbbfe303819d20f7ebb8889bfb7aa60', '4c302ffaacee239105960453f0c59331ce082f32')
        contract_before = New-Contract 'full-BZ direct basis' 'full k/q grid' 'unchanged' 'full-grid owner' 'explicit full-grid weights'
        contract_after = New-Contract 'full-BZ direct basis' 'full k/q grid' 'unchanged' 'full-grid owner' 'explicit full-grid weights'
        classification = 'U1'
        required_change = 'none'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'head-only')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-HEADWING-DIAGNOSTIC-COLLECTIVES'
        formula = 'Diagnostic collectives do not modify head or wing tensors.'
        upstream_symbols = @('diele_func::cal_head', 'diele_func::cal_wing', 'diele_func::test_head', 'diele_func::test_wing')
        qsgw_symbols = @('refresh_headwing')
        upstream_change_ids = @('UP-HEADWING-DIAGNOSTICS-001')
        upstream_commits = @('133a606143e8c9b765277051204c1bc0f59144fe')
        contract_before = New-Contract 'head/wing tensor' 'upstream-defined' 'unchanged' 'global diagnostic collectives' 'unchanged'
        contract_after = New-Contract 'head/wing tensor' 'upstream-defined' 'unchanged' 'Dataset communicator diagnostic collectives' 'unchanged'
        classification = 'U1'
        required_change = 'none'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'head-only', 'head-plus-wing')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-KBLACS-DM-GF'
        formula = 'Distributed D(R) and G(R,tau) equal the serial full-grid transform with single ownership.'
        upstream_symbols = @('get_dmat_cplx_Rs_kblacs_para', 'get_gf_cplx_imagtimes_Rs_kblacs_para', 'G0W0::build_spacetime', 'Exx::build')
        qsgw_symbols = @('run_qsgw', 'collect_sigma_root', 'copy_exx_root')
        upstream_change_ids = @('UP-KBLACS-MF-OWNERSHIP-001')
        upstream_commits = @('318e3e426a3dfac4810d8a5c060021f8713d84dc')
        contract_before = New-Contract 'KS/AO transforms' 'distributed blocks' 'Hartree' 'replicated/restored intermediates' 'unchanged'
        contract_after = New-Contract 'KS/AO transforms' 'distributed blocks' 'Hartree' 'explicit kBLACS ownership' 'unchanged'
        classification = 'U2'
        required_change = 'qsgw_adapter'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'mpi-omp')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-FULLBZ-HEADWING-LIVE'
        formula = 'C_n=C0 U_n and v_n=U_n^dagger v0 U_n feed the full-BZ head/wing at iteration n.'
        upstream_symbols = @('headwing_local_kpoints', 'diele_func::init', 'diele_func::cal_head_full_bz', 'diele_func::cal_wing_full_bz')
        qsgw_symbols = @('refresh_headwing', 'diagonalize_in_reference_basis', 'update_independent_headwing_state')
        upstream_change_ids = @('UP-KPARA-HEADWING-CONTRACT-001')
        upstream_commits = @('a033ec4caec7cec4785b74d1998ec484deedf0b3')
        contract_before = New-Contract 'KS/head-wing basis' 'active k blocks' 'Hartree' 'legacy local-k binding' 'legacy k weights'
        contract_after = New-Contract 'KS/head-wing basis' 'active k blocks' 'Hartree' 'explicit k-parallel ownership' 'full-grid weights'
        classification = 'U2'
        required_change = 'qsgw_adapter'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'head-only', 'head-plus-wing')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-ATOM-BASIS-MAP'
        formula = 'Atom AO counts are authoritative AtomicBasis metadata and match WFC columns.'
        upstream_symbols = @('AtomicBasis::get_atom_nb_map', 'build_gf_Rt_libri_kblacs_para', 'build_dmat_libri_kblacs_para', 'build_gf_libri_kblacs_para')
        qsgw_symbols = @('validate_stage_one_contract')
        upstream_change_ids = @('UP-ATOM-BASIS-MAP-001')
        upstream_commits = @('e238a7615235e604fa5fdc40e0ed7bde8f86be75')
        contract_before = New-Contract 'AO basis' 'atom AO map' 'dimensionless count' 'helper reconstructed' 'unchanged'
        contract_after = New-Contract 'AO basis' 'atom AO map' 'dimensionless count' 'AtomicBasis authoritative' 'unchanged'
        classification = 'U1'
        required_change = 'none'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-COMM-OWNERSHIP'
        formula = 'All collectives for a Dataset operation use Dataset::comm_h.'
        upstream_symbols = @('FT_Vq', 'gather_symmetry_ibz_blocks_collective', 'librpa_build_exx', 'librpa_build_g0w0_sigma')
        qsgw_symbols = @('run_qsgw', 'collective_root_stage')
        upstream_change_ids = @('UP-COMMUNICATOR-OWNERSHIP-001')
        upstream_commits = @('14704c11c735069afd8c23e95a5f087731c3ce56')
        contract_before = New-Contract 'shared operator basis' 'distributed blocks' 'Hartree' 'implicit global communicator' 'unchanged'
        contract_after = New-Contract 'shared operator basis' 'distributed blocks' 'Hartree' 'explicit Dataset communicator' 'unchanged'
        classification = 'U2'
        required_change = 'qsgw_adapter'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'mpi-omp')
        evidence = @('qsgw-rebase-evidence/impact/formula-to-code.md')
    },
    [ordered]@{
        formula_id = 'F-HEAD-TENSOR-READ'
        formula = 'A const view reads the already-computed head tensor without changing numerical state or call order.'
        upstream_symbols = @('diele_func::head', 'diele_func::get_head_matrices')
        qsgw_symbols = @('copy_head_tensor')
        upstream_change_ids = @('U3-HEAD-MATRIX-GETTER-001')
        upstream_commits = @('a033ec4caec7cec4785b74d1998ec484deedf0b3')
        contract_before = New-Contract 'private head Cartesian tensor' '3x3 per frequency' 'upstream dielectric units' 'diele_func private' 'unchanged'
        contract_after = New-Contract 'const head Cartesian tensor view' '3x3 per frequency' 'upstream dielectric units' 'read-only QSGW observer' 'unchanged'
        classification = 'U3'
        required_change = 'approved_shared_change'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = @('qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md')
    }
)

$data['shared_core_hunks'] = @(
    [ordered]@{
        id = 'SHARED-HEAD-GETTER-001'
        path = 'src/core/dielecmodel.h'
        symbols = @('diele_func::head', 'diele_func::get_head_matrices')
        candidate_diff = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.patch'
        upstream_change_id = 'U3-HEAD-MATRIX-GETTER-001'
        classification = 'U3'
        restores_old_implementation = $false
        formula_map_ids = @('F-HEAD-TENSOR-READ')
        benchmark_observer_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        approval_ids = @('APPROVAL-U3-HEAD-MATRIX-GETTER-001')
        status = 'approved'
    }
)

$data['approvals'] = @(
    [ordered]@{
        id = 'APPROVAL-U3-HEAD-MATRIX-GETTER-001'
        type = 'user'
        approved = $true
        scope = 'Only the exact five-line read-only diele_func::get_head_matrices getter in the U3 impact packet and the listed regression observers.'
        approved_by = 'workspace user'
        approved_at = '2026-07-15T16:40:50+08:00'
        evidence_reference = 'qsgw-rebase-evidence/impact/APPROVAL-U3-HEAD-MATRIX-GETTER-001.md'
        candidate_diff = 'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.patch'
        change_ids = @('U3-HEAD-MATRIX-GETTER-001')
        shared_core_hunk_ids = @('SHARED-HEAD-GETTER-001')
        formula_map_ids = @('F-HEAD-TENSOR-READ')
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
    }
)
$data['current_gate'] = 'build-unit-upstream-regressions'
$data['verified_evidence'] = @(
    'qsgw-rebase-evidence/git/status.txt',
    'qsgw-rebase-evidence/git/protected-src-inventory.json',
    'qsgw-rebase-evidence/git/candidate-source.txt',
    $inventoryPath,
    'qsgw-rebase-evidence/impact/formula-to-code.md',
    'qsgw-rebase-evidence/impact/qsgw-source-lane-audit.md',
    'qsgw-rebase-evidence/impact/U3-head-matrix-readonly-getter.md',
    'qsgw-rebase-evidence/impact/APPROVAL-U3-HEAD-MATRIX-GETTER-001.md',
    'qsgw-rebase-evidence/validation/local-precommit-checks.md'
)
$data['open_issues'] = @(
    [ordered]@{
        id = 'ISSUE-GATE0-PROVENANCE'
        class_or_gate = 'build-unit-upstream-regressions'
        blocker = 'No clean candidate executable, remote compiler/library capture, or accepted Si-k444 input manifest exists yet.'
        required = 'Build immutable upstream and candidate commits on the approved remote compute surface.'
    }
)
$data['next_actions'] = @(
    [ordered]@{
        id = 'next-gate0'
        action = 'Build and test immutable upstream and candidate commits on fish, then record exact executable and environment provenance.'
        gate = 'build-unit-upstream-regressions'
        owner = 'rebase operator'
        status = 'ready'
    }
)

$data['planning_state'] = [ordered]@{
    frozen_dirty_source = $false
    clean_candidate_available = $true
    u3_head_matrix_getter = 'approved_exact_five_line_readonly_getter'
    provenance_policy = 'null means missing and must fail validation; no placeholder or synthetic hash is permitted'
    crystal_symmetry_qsgw = 'out_of_scope_not_tested'
}

[System.IO.File]::WriteAllText(
    $Manifest,
    ($data | ConvertTo-Json -Depth 100),
    $utf8
)
