param(
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Manifest = (Join-Path $Repository 'qsgw-rebase-manifest.json')
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$frozenParent = '7e11dd65050666a04361f2d2bd09c7b3aca81c9c'
$upstream = '42d3863c1d865194d382a085851d1e2e8a39764f'
$noSymCandidate = 'c27482016f70ece5a0e5ccad7199d93ac3f6ebf5'
$branch = 'codex/qsgw-symmetry-no-headwing-42d-20260720'

function Set-Property {
    param($Object, [string]$Name, $Value)
    if ($Object.PSObject.Properties.Name -contains $Name) {
        $Object.$Name = $Value
    }
    else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

$data = Get-Content -Raw -Encoding UTF8 -LiteralPath $Manifest |
    ConvertFrom-Json

$data.schema_version = '1.0'
$data.repository.path = $Repository.Replace('\', '/')
$data.repository.worktree = $Repository.Replace('\', '/')
$data.repository.branch = $branch
$data.repository.commits.branch_head.hash = $frozenParent
$data.repository.commits.branch_head.reference =
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
$data.repository.commits.rebase_head.hash = $frozenParent
$data.repository.commits.rebase_head.reference =
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
$data.repository.commits.upstream_new.hash = $upstream
$data.repository.commits.upstream_new.reference =
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'

$data.provenance.source.hash = $frozenParent
$data.provenance.source.reference =
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
$data.provenance.executable.path = $null
$data.provenance.executable.sha256 = $null
$data.provenance.executable.source_commit = $frozenParent
$data.provenance.executable.reference =
    'qsgw-rebase-evidence/remote/dongfang-symmetry-k888-20260720/fish-full-ctest-v2'
$data.provenance.dataset.id = 'si-k888-symmetry-inventory-supporting'
$data.provenance.dataset.path =
    '/data/home/df_iopcas_bhj/ai-runs/si-qsgw-band0-k888-sym-shrink-headwing-iter1-20260514-173351'
$data.provenance.dataset.sha256 = $null
$data.provenance.dataset.reference =
    'qsgw-rebase-evidence/remote/dongfang-symmetry-k888-20260720/dongfang-inventory-v2-failed'

$data.gate_sequence = @(
    'build-unit-upstream-regressions',
    'g0w0-upstream-vs-rebased',
    'qsgw-iter0-vs-upstream-g0w0',
    'solid-qsgw-no-mixing-old-vs-new',
    'head-only',
    'head-plus-wing',
    'mixing',
    'hartree',
    'qsgw-band',
    'fhi-aims',
    'abacus',
    'mpi-omp'
)

foreach ($benchmark in @($data.benchmarks)) {
    switch ($benchmark.id) {
        'head-only' {
            $benchmark.description =
                'QSGW head-only request must fail during input validation'
            $benchmark.observer_types = @('head_only')
            $benchmark.required_artifacts = @(
                'head-only-result',
                'g0w0-limit',
                'structural-invariants',
                'per-iteration-replay',
                'independent-comparison',
                'parser-rejection',
                'runtime-guard',
                'unchanged-g0w0-headwing-regression'
            )
            $benchmark.oracle_reliability = 'unreliable'
            $benchmark.oracle_reference =
                'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
        }
        'head-plus-wing' {
            $benchmark.description =
                'QSGW head-plus-wing request must fail during input validation'
            $benchmark.observer_types = @('head_plus_wing')
            $benchmark.required_artifacts = @(
                'head-wing-result',
                'g0w0-limit',
                'structural-invariants',
                'per-iteration-replay',
                'independent-comparison',
                'parser-rejection',
                'runtime-guard',
                'unchanged-g0w0-headwing-regression'
            )
            $benchmark.oracle_reliability = 'unreliable'
            $benchmark.oracle_reference =
                'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
        }
        'hartree' {
            $benchmark.description =
                'Two-round Hartree-on no-symmetry and ABACUS symmetry gate using live density'
            $benchmark.required_artifacts = @(
                'hartree-result',
                'hartree-comparison',
                'delta-rho-charge',
                'delta-vh-iter0-zero',
                'q0-g0-convention',
                'unit-validation',
                'grid-band-hermiticity',
                'legacy-new-two-round-comparison'
            )
        }
        'qsgw-band' {
            $benchmark.description =
                'Head-wing-off grid AO operator to BvK/Fourier fixed-band projection'
            $benchmark.required_artifacts = @(
                'band-data',
                'band-comparison',
                'grid-ao-operator',
                'full-bz-fourier-diagnostics',
                'symmetry-restoration-diagnostics',
                'band-component-comparison'
            )
        }
        'symmetry' {
            $benchmark.gate = 'mpi-omp'
            $benchmark.description =
                'ABACUS Si 29-to-512 symmetry-on two-round three-way acceptance'
            $benchmark.required_artifacts = @(
                'symmetry-result',
                'symmetry-invariants',
                'mapping-validation',
                'rotation-phase-time-reversal-validation',
                'legacy-symmetry-oracle',
                'full-bz-symmetry-component-comparison',
                'projector-comparison'
            )
        }
    }
}

foreach ($benchmark in @($data.benchmarks)) {
    if ($benchmark.id -in @(
            'g0w0-upstream-vs-rebased',
            'qsgw-iter0-vs-upstream')) {
        $benchmark.definition.side_a.source.hash = $upstream
        $benchmark.definition.side_a.source.reference =
            'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
        $benchmark.definition.side_a.executable.source_commit = $upstream
        $benchmark.definition.side_b.source.hash = $frozenParent
        $benchmark.definition.side_b.source.reference =
            $data.provenance.source.reference
        $benchmark.definition.side_b.executable.path =
            $data.provenance.executable.path
        $benchmark.definition.side_b.executable.sha256 =
            $data.provenance.executable.sha256
        $benchmark.definition.side_b.executable.source_commit = $frozenParent
        $benchmark.definition.side_b.executable.reference =
            $data.provenance.executable.reference
    }
    elseif ($benchmark.id -eq 'solid-qsgw-no-mixing') {
        $benchmark.definition.side_b.source.hash = $frozenParent
        $benchmark.definition.side_b.source.reference =
            $data.provenance.source.reference
        $benchmark.definition.side_b.executable.path =
            $data.provenance.executable.path
        $benchmark.definition.side_b.executable.sha256 =
            $data.provenance.executable.sha256
        $benchmark.definition.side_b.executable.source_commit = $frozenParent
        $benchmark.definition.side_b.executable.reference =
            $data.provenance.executable.reference
    }
}
foreach ($pair in @($data.control_pairs)) {
    foreach ($side in @($pair.a, $pair.b)) {
        $side.source.hash = $frozenParent
        $side.source.reference = $data.provenance.source.reference
        $side.executable.source_commit = $frozenParent
    }
}
$data.upstream_inventory.new_commit = $upstream

$data.current_gate = 'build-unit-upstream-regressions'
$verified = @($data.verified_evidence) + @(
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md',
    'qsgw-rebase-evidence/remote/dongfang-symmetry-k888-20260720/fish-full-ctest-v2',
    'qsgw-rebase-evidence/remote/dongfang-symmetry-k888-20260720/dongfang-inventory-v2-failed'
)
$data.verified_evidence = @($verified | Select-Object -Unique)

$data.open_issues = @(
    [pscustomobject]@{
        id = 'ISSUE-QSGW-HEADWING-REACHABLE'
        class_or_gate = 'qsgw-headwing-fail-fast'
        blocker = 'QSGW parser and runtime still execute iterative head/wing.'
        required = 'Reject QSGW replace_w_head/use_pyatb and remove runtime reachability without changing G0W0.'
    },
    [pscustomobject]@{
        id = 'ISSUE-OBSOLETE-U3-GETTER'
        class_or_gate = 'protected-diff'
        blocker = 'The approved shared head getter is used only by excluded QSGW head/wing code.'
        required = 'Remove the QSGW consumer and getter so shared numerical diff is zero.'
    },
    [pscustomobject]@{
        id = 'ISSUE-BAND-NOT-FOURIER-WIRED'
        class_or_gate = 'qsgw-band-operator-fourier'
        blocker = 'The current qsgw_band loop separately evaluates band EXX/Sigma; operator_fourier is only called by headwing_update.'
        required = 'Project the converged grid AO/real-space operator to immutable mf0_band.'
    },
    [pscustomobject]@{
        id = 'ISSUE-SYMMETRY-ORACLE'
        class_or_gate = 'abacus-symmetry-two-round'
        blocker = 'No merge-before symmetry two-round oracle or three-way full-BZ comparison is frozen.'
        required = 'Regenerate legacy oracle and compare two rounds per component, with miniter5 recommended for acceptance.'
    },
    [pscustomobject]@{
        id = 'ISSUE-HARTREE-E2E'
        class_or_gate = 'hartree-no-symmetry-and-symmetry'
        blocker = 'Hartree unit tests exist but no accepted two-round end-to-end gate exists.'
        required = 'Validate live density, charge, units, q=0 convention, Hermiticity, and legacy equivalence.'
    },
    [pscustomobject]@{
        id = 'ISSUE-REGRESSION-DANGLING'
        class_or_gate = 'formal-regression'
        blocker = 'QSGW testsuite.xml entries reference missing testcase directories.'
        required = 'Commit real ABACUS no-sym, ABACUS symmetry, and FHI-aims two-round cases with legacy references.'
    }
)
$data.next_actions = @(
    [pscustomobject]@{
        id = 'next-qsgw-headwing-fail-fast'
        action = 'Make every QSGW head/wing request fail fast, remove the obsolete shared getter, then build and run the complete test suite.'
        gate = 'build-unit-upstream-regressions'
        owner = 'rebase operator'
        status = 'ready'
    }
)

$data.planning_state.frozen_dirty_source = $true
$data.planning_state.clean_candidate_available = $false
$data.planning_state.u3_head_matrix_getter =
    'approved_but_obsolete_remove_pending'
$data.planning_state.crystal_symmetry_qsgw =
    'required_not_yet_numerically_accepted'
Set-Property -Object $data.planning_state -Name 'qsgw_iterative_headwing' -Value 'unsupported_fail_fast_pending'
Set-Property -Object $data.planning_state -Name 'qsgw_band_operator_fourier' -Value 'not_wired'
Set-Property -Object $data.planning_state -Name 'formal_qsgw_regression' -Value 'dangling_entries_no_committed_cases'

Set-Property $data 'revised_scope' ([pscustomobject]@{
    effective_date = '2026-07-20'
    freeze_parent = $frozenParent
    upstream_base = $upstream
    no_sym_candidate = $noSymCandidate
    branch = $branch
    supports_target = @(
        'full_bz_no_crystal_symmetry',
        'abacus_ibz_crystal_symmetry',
        'linear_fixed_basis_hamiltonian_mixing_beta_0.2',
        'hartree_live_density',
        'headwing_off_qsgw_band_operator_fourier'
    )
    explicitly_unsupported = @(
        'qsgw_iterative_headwing',
        'unvalidated_fhi_aims_symmetry'
    )
    audit = 'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'
})

[System.IO.File]::WriteAllText(
    $Manifest,
    ($data | ConvertTo-Json -Depth 100) + [Environment]::NewLine,
    $utf8
)
