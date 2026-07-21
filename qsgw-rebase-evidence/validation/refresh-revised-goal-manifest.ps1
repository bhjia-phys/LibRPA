param(
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Manifest = (Join-Path $Repository 'qsgw-rebase-manifest.json'),
    [string]$CandidateSourceCommit = '',
    [string]$Gate0Evidence = ''
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$frozenParent = 'b7273e13c77d5ea781f192cea3c4201710b6f9fa'
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

function Invoke-GitLines {
    param([string[]]$GitArgs)

    $lines = @(& git -C $Repository @GitArgs)
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
    return @($lines | Where-Object { $_ -ne $null -and $_.Trim().Length -gt 0 })
}

function Read-KeyValueFile {
    param([string]$Path)

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([^=]+)=(.*)$') {
            $values[$matches[1]] = $matches[2]
        }
    }
    return $values
}

function Assert-ChecksumManifest {
    param([string]$Root, [string]$ChecksumFile)

    $rootFull = [IO.Path]::GetFullPath($Root)
    foreach ($line in Get-Content -LiteralPath $ChecksumFile) {
        if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
            throw "Malformed checksum row: $line"
        }
        $expected = $matches[1]
        $relative = $matches[2]
        if ($relative.StartsWith('./')) {
            $relative = $relative.Substring(2)
        }
        $path = [IO.Path]::GetFullPath((Join-Path $rootFull $relative))
        if (-not $path.StartsWith(
                $rootFull + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase)) {
            throw "Checksum path escapes Gate 0 evidence root: $relative"
        }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Gate 0 checksum target is missing: $relative"
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "Gate 0 checksum mismatch for ${relative}: $actual"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($CandidateSourceCommit)) {
    $candidateSource = @(Invoke-GitLines @('rev-parse', 'HEAD'))[0]
}
else {
    $candidateSource = $CandidateSourceCommit.Trim().ToLowerInvariant()
}
if ($candidateSource -notmatch '^[0-9a-f]{40}$') {
    throw "Candidate source commit is not a full Git SHA: $candidateSource"
}
$verifiedCandidate = @(
    Invoke-GitLines @('rev-parse', '--verify', "$candidateSource`^{commit}")
)[0]
if ($verifiedCandidate -ne $candidateSource) {
    throw "Candidate source did not resolve to the requested commit: $verifiedCandidate"
}
$null = Invoke-GitLines @(
    'merge-base', '--is-ancestor', $frozenParent, $candidateSource
)
$hasCleanCandidate = $candidateSource -ne $frozenParent

if ([string]::IsNullOrWhiteSpace($Gate0Evidence)) {
    $Gate0Evidence = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate0-current-20260721\66bfe1cf-v1'
    )
}
$gate0 = $null
$gate0Reference =
    'qsgw-rebase-evidence/remote/fish-gate0-current-20260721/66bfe1cf-v1/PROVENANCE.txt'
if (Test-Path -LiteralPath $Gate0Evidence -PathType Container) {
    $green = Join-Path $Gate0Evidence 'GREEN_CONFIRMED'
    $failed = Join-Path $Gate0Evidence 'FAILED'
    $provenancePath = Join-Path $Gate0Evidence 'PROVENANCE.txt'
    $checksumPath = Join-Path $Gate0Evidence 'OUTPUT_SHA256SUMS.txt'
    if (-not (Test-Path -LiteralPath $green -PathType Leaf) -or
        (Test-Path -LiteralPath $failed) -or
        -not (Test-Path -LiteralPath $provenancePath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $checksumPath -PathType Leaf)) {
        throw 'Gate 0 evidence is incomplete or contains FAILED'
    }
    Assert-ChecksumManifest -Root $Gate0Evidence -ChecksumFile $checksumPath
    $gate0 = Read-KeyValueFile -Path $provenancePath
    $expectedGate0 = @{
        gate = 'fish_gate0_current_v1'
        acceptance = 'true'
        upstream_commit = $upstream
        candidate_commit = $candidateSource
        upstream_test_count = '39'
        candidate_test_count = '63'
        candidate_focused_test_count = '10'
        failed_test_count = '0'
        not_run_test_count = '0'
        protected_diff = 'empty'
    }
    foreach ($key in $expectedGate0.Keys) {
        if ($gate0[$key] -ne $expectedGate0[$key]) {
            throw "Gate 0 provenance mismatch for ${key}: $($gate0[$key])"
        }
    }
    $runnerPath = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate0-current-20260721\run_fish_gate0_current_v1.sh'
    )
    $runnerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $runnerPath).Hash.ToLowerInvariant()
    if ($runnerHash -ne $gate0.runner_sha256) {
        throw "Gate 0 runner hash mismatch: $runnerHash"
    }
    foreach ($side in @('upstream', 'candidate')) {
        $hashFile = Join-Path $Gate0Evidence "${side}-executable.sha256"
        $hashValue = ((Get-Content -LiteralPath $hashFile -Raw).Trim() -split '\s+')[0]
        if ($hashValue -ne $gate0["${side}_executable_sha256"]) {
            throw "Gate 0 ${side} executable hash record mismatch"
        }
    }
}

function Get-CommitPaths {
    param([string]$Commit)

    return @(Invoke-GitLines @(
        'diff-tree', '--no-commit-id', '--name-only', '-r', $Commit
    ))
}

function New-SemanticAudit {
    param(
        [string]$Classification,
        [string[]]$ChangedFields,
        [string]$Evidence
    )

    $default = if ($Classification -eq 'U0') { 'not_applicable' } else { 'unchanged' }
    $audit = [ordered]@{
        api = $default
        matrix_layout = $default
        basis = $default
        units = $default
        normalization = $default
        mpi_ownership = $default
        k_point_weights = $default
        occupations = $default
        call_ordering = $default
        symmetry = $default
        evidence = $Evidence
    }
    foreach ($field in @($ChangedFields)) {
        if (-not $audit.Contains($field)) {
            throw "Unknown semantic audit field: $field"
        }
        $audit[$field] = 'changed'
    }
    return [pscustomobject]$audit
}

function New-FormulaContract {
    param(
        [string]$Basis,
        [string]$Shape,
        [string]$Units,
        [string]$Ownership,
        [string]$Normalization
    )

    return [pscustomobject][ordered]@{
        basis = $Basis
        shape = $Shape
        units = $Units
        ownership = $Ownership
        normalization = $Normalization
    }
}

function Set-HashedArtifact {
    param(
        $Record,
        [string]$RelativePath,
        [string]$Reference
    )

    $absolutePath = Join-Path $Repository ($RelativePath.Replace('/', '\'))
    $Record.path = $RelativePath
    $Record.sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $absolutePath).Hash.ToLowerInvariant()
    $Record.reference = $Reference
}

$data = Get-Content -Raw -Encoding UTF8 -LiteralPath $Manifest |
    ConvertFrom-Json

$data.schema_version = '1.0'
$data.repository.path = $Repository.Replace('\', '/')
$data.repository.worktree = $Repository.Replace('\', '/')
$data.repository.branch = $branch
$data.repository.commits.branch_head.hash = $candidateSource
$data.repository.commits.branch_head.reference =
    'qsgw-rebase-evidence/validation/local-audit-20260721.md'
$data.repository.commits.rebase_head.hash = $candidateSource
$data.repository.commits.rebase_head.reference =
    'qsgw-rebase-evidence/validation/local-audit-20260721.md'
$data.repository.commits.upstream_new.hash = $upstream
$data.repository.commits.upstream_new.reference =
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md'

$data.provenance.source.hash = $candidateSource
$data.provenance.source.reference =
    'qsgw-rebase-evidence/validation/local-audit-20260721.md'
$data.provenance.executable.path = if ($gate0) {
    $gate0.candidate_executable
} else {
    $null
}
$data.provenance.executable.sha256 = if ($gate0) {
    $gate0.candidate_executable_sha256
} else {
    $null
}
$data.provenance.executable.source_commit = $candidateSource
$data.provenance.executable.reference = if ($gate0) {
    $gate0Reference
} else {
    'pending: clean-candidate fish Gate 0'
}
$data.provenance.dataset.id = 'si-k444-symmetry-gate-a-v3-supporting'
$data.provenance.dataset.path =
    '/home/bhj/ai-runs/librpa-qsgw-gate-a-symmetry-bundle-20260720-v3/dataset'
$data.provenance.dataset.sha256 =
    '869f4fd922dc1085af2cc02644f2a65e5b462237fde6428eed5a46143e833690'
$data.provenance.dataset.reference =
    'qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/run_gate_a1_symmetry_nomix_miniter2_v6.sh'

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
        if ($gate0) {
            $benchmark.definition.side_a.executable.path =
                $gate0.upstream_executable
            $benchmark.definition.side_a.executable.sha256 =
                $gate0.upstream_executable_sha256
            $benchmark.definition.side_a.executable.reference =
                $gate0Reference
        }
        $benchmark.definition.side_b.source.hash = $candidateSource
        $benchmark.definition.side_b.source.reference =
            $data.provenance.source.reference
        $benchmark.definition.side_b.executable.path =
            $data.provenance.executable.path
        $benchmark.definition.side_b.executable.sha256 =
            $data.provenance.executable.sha256
        $benchmark.definition.side_b.executable.source_commit = $candidateSource
        $benchmark.definition.side_b.executable.reference =
            $data.provenance.executable.reference
    }
    elseif ($benchmark.id -eq 'solid-qsgw-no-mixing') {
        $benchmark.definition.side_b.source.hash = $candidateSource
        $benchmark.definition.side_b.source.reference =
            $data.provenance.source.reference
        $benchmark.definition.side_b.executable.path =
            $data.provenance.executable.path
        $benchmark.definition.side_b.executable.sha256 =
            $data.provenance.executable.sha256
        $benchmark.definition.side_b.executable.source_commit = $candidateSource
        $benchmark.definition.side_b.executable.reference =
            $data.provenance.executable.reference
    }
}
foreach ($pair in @($data.control_pairs)) {
    foreach ($side in @($pair.a, $pair.b)) {
        $side.source.hash = $candidateSource
        $side.source.reference = $data.provenance.source.reference
        $side.executable.source_commit = $candidateSource
        if ($gate0) {
            $side.executable.path = $data.provenance.executable.path
            $side.executable.sha256 = $data.provenance.executable.sha256
            $side.executable.reference = $data.provenance.executable.reference
        }
    }
}
if ($gate0) {
    $buildBenchmark = @(
        $data.benchmarks | Where-Object { $_.id -eq 'build-upstream-regressions' }
    )[0]
    $artifactSpecs = @(
        @('build-log', 'candidate-build.stdout'),
        @('unit-results', 'candidate-ctest.xml'),
        @('upstream-regression-results', 'upstream-ctest.xml'),
        @('focused-qsgw-results', 'candidate-focused-ctest.xml'),
        @('python-qsgw-results', 'candidate-python-tests.stdout'),
        @('runtime-parameter-doc', 'candidate-runtime-parameters.md'),
        @('protected-diff', 'protected-diff.patch'),
        @('gate0-provenance', 'PROVENANCE.txt'),
        @('gate0-checksum-manifest', 'OUTPUT_SHA256SUMS.txt')
    )
    $buildArtifacts = @()
    foreach ($spec in $artifactSpecs) {
        $relative =
            'qsgw-rebase-evidence/remote/fish-gate0-current-20260721/66bfe1cf-v1/' +
            $spec[1]
        $record = [pscustomobject][ordered]@{
            id = $spec[0]
            path = $null
            sha256 = $null
            reference = $null
        }
        Set-HashedArtifact -Record $record -RelativePath $relative `
            -Reference $gate0Reference
        $buildArtifacts += $record
    }
    $buildBenchmark.status = 'accepted'
    $buildBenchmark.artifacts = @($buildArtifacts)
}
$data.upstream_inventory.new_commit = $upstream

$refreshAuditEarly =
    'qsgw-rebase-evidence/git/upstream-refresh-1376ee4f-to-95c4c080.md'
$refreshAuditLate =
    'qsgw-rebase-evidence/git/upstream-refresh-95c4c080-to-42d3863c.md'
$refreshSpecs = @(
    [pscustomobject]@{
        id = 'UP-BN-GW-HEADWING-REGRESSION-001'
        commit = '293dacc5ff8a186d39f18d818a5d9c66d45f7968'
        classification = 'U0'
        summary = 'Adds the BN symmetry+kpara GW head/wing regression outside src.'
        semantic_changes = @()
        symbols = @()
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-EMPTY-RANK-WQ-RESTORE-TEST-001'
        commit = 'b66d528353aab6283fcdd3d0b5765dfd3816bd50'
        classification = 'U1'
        summary = 'Extends protected shared tests for empty-rank symmetry W(q) restoration.'
        semantic_changes = @()
        symbols = @('restore_symmetry_dense_wq_map')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-BN-RPA-KPARA-REGRESSION-001'
        commit = 'b51273861f541985c2419c81ea1a031c668ddcc7'
        classification = 'U0'
        summary = 'Enables k parallelism in the BN symmetry RPA regression outside src.'
        semantic_changes = @()
        symbols = @()
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-CS-WORLD-K-REDISTRIBUTION-001'
        commit = '2b34a92f13133ed587b6fa6d5f8946c4c6d74eed'
        classification = 'U2'
        summary = 'Moves head/wing Cs Fourier and redistribution work to explicit world and k ownership.'
        semantic_changes = @('api', 'matrix_layout', 'mpi_ownership', 'call_ordering', 'symmetry')
        symbols = @('read_Cs', 'initialize_ds_Cs', 'diele_func::cal_head_symmetric')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-HEADWING-TEST-DATA-001'
        commit = 'c3563afd82538d1f7283808470bdcf7f78a64a87'
        classification = 'U1'
        summary = 'Updates protected Dataset/head-wing tests and one regression input.'
        semantic_changes = @()
        symbols = @('test_dataset', 'test_rpa_headwing')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-CS-ROTATION-BLOCK-128-001'
        commit = 'be966e250ac5162774291ec33f4c608c9c25d390'
        classification = 'U2'
        summary = 'Changes the distributed rectangular block layout used for Cs symmetry rotation.'
        semantic_changes = @('matrix_layout', 'mpi_ownership', 'symmetry')
        symbols = @('rotate_Cs_nao2mnk_kblacs')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-WFC-KCOMM-2D-READ-001'
        commit = '3ffe9ab8d35d8307ecc7ee20470118d6763fab89'
        classification = 'U2'
        summary = 'Changes WFC reader, Dataset, EXX, and GW APIs to k-communicator 2D block ownership.'
        semantic_changes = @('api', 'matrix_layout', 'mpi_ownership', 'call_ordering')
        symbols = @('read_eigenvectors', 'Dataset', 'Exx::build', 'G0W0::build_spacetime')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-SYM-WFC-LOCAL-ROTATION-001'
        commit = '2df7b08bb0700147cded4097ddf35de12519d0d2'
        classification = 'U2'
        summary = 'Restores symmetry rotation from the correct locally owned WFC block.'
        semantic_changes = @('matrix_layout', 'mpi_ownership', 'symmetry')
        symbols = @('diele_func::cal_head_symmetric', 'diele_func::cal_wing_symmetric')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-WFC-RECTANGULAR-REDISTRIBUTION-001'
        commit = '12f43c38091db37f13c49965fffabd7b77acac74'
        classification = 'U2'
        summary = 'Introduces capped rectangular WFC redistribution descriptors after input.'
        semantic_changes = @('api', 'matrix_layout', 'mpi_ownership', 'call_ordering')
        symbols = @('Dataset', 'MeanField', 'Exx::build', 'G0W0::build_spacetime')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-WFC-VELOCITY-FULLBZ-PAIR-001'
        commit = '14261e285ec3b2db2db1d0445a3d233a37f4c0c7'
        classification = 'U2'
        summary = 'Pairs full-BZ WFC and velocity members under the same symmetry transform.'
        semantic_changes = @('basis', 'mpi_ownership', 'call_ordering', 'symmetry')
        symbols = @('diele_func::cal_head_symmetric', 'diele_func::cal_wing_symmetric')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-WFC-BLOCK-CYCLIC-LAYOUT-001'
        commit = '95c4c08009aa6752a6d386289abe1fb4358489ca'
        classification = 'U2'
        summary = 'Accepts valid WFC block-cyclic layouts instead of requiring a dense-root shape.'
        semantic_changes = @('api', 'matrix_layout', 'mpi_ownership')
        symbols = @('validate_wfc_layout', 'Dataset', 'MeanField', 'Exx::build', 'G0W0::build_spacetime')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        id = 'UP-REAL-DENSE-WC-001'
        commit = 'eefa0682ad5e6796237a80edaabcbc75caab91c1'
        classification = 'U2'
        summary = 'Adds an optional real-valued dense Wc CT/FT representation and shared GW path.'
        semantic_changes = @('api', 'matrix_layout', 'call_ordering')
        symbols = @('FT_Wc_freq_q_into', 'CT_FT_Wc_freq_q_real', 'G0W0::build_spacetime')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        id = 'UP-LIH-REAL-WC-REGRESSION-001'
        commit = 'deadd00853920e12653c9b336f8957a469d34efb'
        classification = 'U0'
        summary = 'Enables the real dense Wc developer option in the LiH regression input.'
        semantic_changes = @()
        symbols = @()
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        id = 'UP-DENSE-WC-MEMORY-LIFETIME-001'
        commit = '9bb5fa4bc86b9bc3efca545bac2527d93fcf16b7'
        classification = 'U1'
        summary = 'Releases each dense Wc time-point source map after LibRI tensor preparation.'
        semantic_changes = @('call_ordering')
        symbols = @('prepare_dense_wc_libri')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        id = 'UP-REAL-WC-RESIDUAL-WARN-001'
        commit = '51b9e08ef91a5e75edc2f5d443d28d5402e41f5e'
        classification = 'U1'
        summary = 'Warns on a finite real-Wc imaginary residual instead of aborting before projection.'
        semantic_changes = @()
        symbols = @('validate_dense_wc_real_residual')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        id = 'UP-EPSILON-TEST-SPLIT-001'
        commit = '42d3863c1d865194d382a085851d1e2e8a39764f'
        classification = 'U1'
        summary = 'Splits epsilon tests from head-wing tests inside the protected src tree.'
        semantic_changes = @()
        symbols = @('test_epsilon', 'test_rpa_headwing')
        evidence = $refreshAuditLate
    }
)

$retiredChangeIds = @('U3-HEAD-MATRIX-GETTER-001')
$retiredFormulaIds = @('F-HEAD-TENSOR-READ')
$retiredSharedHunkIds = @('SHARED-HEAD-GETTER-001')
$retiredApprovalIds = @('APPROVAL-U3-HEAD-MATRIX-GETTER-001')
$refreshIds = @($refreshSpecs | ForEach-Object { $_.id })
$refreshCommits = @($refreshSpecs | ForEach-Object { $_.commit })
$generatedChanges = @($refreshSpecs | ForEach-Object {
    $spec = $_
    $classification = $spec.classification
    $effectDomain = switch ($classification) {
        'U0' { 'none' }
        'U1' { 'upstream_g0w0_inherited' }
        'U2' { 'qsgw_adapter' }
        default { throw "Unsupported refresh classification: $classification" }
    }
    $disposition = if ($classification -eq 'U2') {
        'adapt_qsgw_only'
    }
    else {
        'accept_upstream'
    }
    [string[]]$benchmarks = if ($classification -eq 'U0') {
        'build-upstream-regressions'
    }
    else {
        'g0w0-upstream-vs-rebased'
        'qsgw-iter0-vs-upstream'
    }
    [string[]]$adapterBenchmarks = if ($classification -eq 'U2') {
        'qsgw-iter0-vs-upstream'
    }
    else {
        [string[]]@()
    }

    $record = [pscustomobject][ordered]@{
        id = $spec.id
        commit = $spec.commit
        commit_reference = $spec.evidence
        classification = $classification
        effect_domain = $effectDomain
        changed_paths = @(Get-CommitPaths $spec.commit)
        summary = $spec.summary
        disposition = $disposition
        semantic_audit = New-SemanticAudit `
            -Classification $classification `
            -ChangedFields $spec.semantic_changes `
            -Evidence $spec.evidence
        benchmark_observer_ids = @()
        qsgw_adapter_observer_ids = @()
        approval_ids = @()
    }
    $record.benchmark_observer_ids = @($benchmarks)
    if ($classification -eq 'U2') {
        $record.qsgw_adapter_observer_ids = @($adapterBenchmarks)
    }
    $record
})
$data.upstream_changes = @(
    @($data.upstream_changes | Where-Object {
        $_.id -notin $refreshIds -and $_.id -notin $retiredChangeIds
    }) +
    $generatedChanges
)

$formulaSpecs = @(
    [pscustomobject]@{
        formula_id = 'F-REFRESH-HEADWING-TEST-OBSERVERS'
        formula = 'Protected head-wing test refactors preserve the shared restoration formula.'
        upstream_symbols = @('restore_symmetry_dense_wq_map', 'test_rpa_headwing')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-EMPTY-RANK-WQ-RESTORE-TEST-001', 'UP-HEADWING-TEST-DATA-001')
        classification = 'U1'
        before = New-FormulaContract 'shared head-wing basis' 'unchanged' 'dimensionless' 'upstream test communicator' 'unchanged'
        after = New-FormulaContract 'shared head-wing basis' 'unchanged' 'dimensionless' 'upstream test communicator' 'unchanged'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        formula_id = 'F-CS-HEADWING-DISTRIBUTION'
        formula = 'Distributed Cs rotation and Fourier redistribution equal the full matrix transform with one owner per k block.'
        upstream_symbols = @('read_Cs', 'initialize_ds_Cs', 'rotate_Cs_nao2mnk_kblacs')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-CS-WORLD-K-REDISTRIBUTION-001', 'UP-CS-ROTATION-BLOCK-128-001')
        classification = 'U2'
        before = New-FormulaContract 'AO-product basis' 'legacy distributed Cs blocks' 'basis-dependent coefficient' 'local/subcommunicator ownership' 'unchanged'
        after = New-FormulaContract 'AO-product basis' '128-column rectangular Cs blocks' 'basis-dependent coefficient' 'world Fourier plus explicit k ownership' 'unchanged'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'mpi-omp')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        formula_id = 'F-WFC-DISTRIBUTED-READ-LAYOUT'
        formula = 'A block-cyclic WFC represents the same KS eigenvectors after reader and Dataset redistribution.'
        upstream_symbols = @('read_eigenvectors', 'Dataset', 'MeanField', 'Exx::build', 'G0W0::build_spacetime')
        qsgw_symbols = @('run_qsgw', 'QsgwState')
        change_ids = @('UP-WFC-KCOMM-2D-READ-001', 'UP-WFC-RECTANGULAR-REDISTRIBUTION-001', 'UP-WFC-BLOCK-CYCLIC-LAYOUT-001')
        classification = 'U2'
        before = New-FormulaContract 'KS eigenvector basis' 'dense-root or legacy distributed layout' 'dimensionless' 'legacy reader/global ownership' 'orthonormal columns'
        after = New-FormulaContract 'KS eigenvector basis' 'valid capped rectangular block-cyclic layout' 'dimensionless' 'explicit k-communicator ownership' 'orthonormal columns'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'mpi-omp')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        formula_id = 'F-SYMMETRY-WFC-VELOCITY-PAIRING'
        formula = 'A full-BZ symmetry member uses its matched WFC and velocity under the same rotation and time-reversal operation.'
        upstream_symbols = @('diele_func::cal_head_symmetric', 'diele_func::cal_wing_symmetric')
        qsgw_symbols = @('run_qsgw', 'QsgwState')
        change_ids = @('UP-SYM-WFC-LOCAL-ROTATION-001', 'UP-WFC-VELOCITY-FULLBZ-PAIR-001')
        classification = 'U2'
        before = New-FormulaContract 'IBZ/full-BZ KS basis' 'independently selected WFC and velocity members' 'Hartree and velocity units' 'partially implicit local ownership' 'upstream symmetry normalization'
        after = New-FormulaContract 'matched full-BZ KS basis' 'paired WFC and velocity symmetry members' 'Hartree and velocity units' 'explicit local member ownership' 'upstream symmetry normalization'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream', 'mpi-omp')
        evidence = $refreshAuditEarly
    },
    [pscustomobject]@{
        formula_id = 'F-REAL-DENSE-WC-CTFT'
        formula = 'For scalar Wc with a bounded imaginary residual, real dense CT/FT storage represents the real projection of the complex path.'
        upstream_symbols = @('FT_Wc_freq_q_into', 'CT_FT_Wc_freq_q_real', 'G0W0::build_spacetime')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-REAL-DENSE-WC-001')
        classification = 'U2'
        before = New-FormulaContract 'auxiliary product basis' 'complex dense Wc(R,tau)' 'screened interaction' 'distributed dense matrices' '1/Nk Fourier normalization'
        after = New-FormulaContract 'auxiliary product basis' 'optional real dense Wc(R,tau)' 'screened interaction' 'distributed dense matrices after collective residual check' '1/Nk Fourier normalization'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        formula_id = 'F-DENSE-WC-LIFETIME'
        formula = 'Releasing a dense Wc source map after LibRI tensor preparation does not change the subsequent correlation self-energy.'
        upstream_symbols = @('prepare_dense_wc_libri', 'G0W0::build_spacetime')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-DENSE-WC-MEMORY-LIFETIME-001')
        classification = 'U1'
        before = New-FormulaContract 'auxiliary product basis' 'dense Wc per time point' 'screened interaction' 'source maps retained after tensor preparation' 'unchanged'
        after = New-FormulaContract 'auxiliary product basis' 'dense Wc per time point' 'screened interaction' 'source map released after tensor preparation' 'unchanged'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        formula_id = 'F-REAL-WC-RESIDUAL-POLICY'
        formula = 'The real dense Wc projection and its residual diagnostic do not alter the projected matrix formula.'
        upstream_symbols = @('validate_dense_wc_real_residual')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-REAL-WC-RESIDUAL-WARN-001')
        classification = 'U1'
        before = New-FormulaContract 'auxiliary product basis' 'real projection of dense Wc' 'screened interaction' 'collective residual diagnostic' 'unchanged'
        after = New-FormulaContract 'auxiliary product basis' 'real projection of dense Wc' 'screened interaction' 'collective residual diagnostic' 'unchanged'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditLate
    },
    [pscustomobject]@{
        formula_id = 'F-EPSILON-TEST-SPLIT'
        formula = 'Moving epsilon test cases between test translation units leaves the shared epsilon implementation unchanged.'
        upstream_symbols = @('test_epsilon', 'test_rpa_headwing')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-EPSILON-TEST-SPLIT-001')
        classification = 'U1'
        before = New-FormulaContract 'epsilon test basis' 'same test matrices' 'dimensionless' 'same test communicator' 'unchanged'
        after = New-FormulaContract 'epsilon test basis' 'same test matrices' 'dimensionless' 'same test communicator' 'unchanged'
        benchmark_ids = @('g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditLate
    }
)

$formulaIds = @($formulaSpecs | ForEach-Object { $_.formula_id })
$generatedFormulaRows = @($formulaSpecs | ForEach-Object {
    $formula = $_
    $commits = @($formula.change_ids | ForEach-Object {
        $changeId = $_
        @($refreshSpecs | Where-Object { $_.id -eq $changeId })[0].commit
    })
    [pscustomobject][ordered]@{
        formula_id = $formula.formula_id
        formula = $formula.formula
        upstream_symbols = $formula.upstream_symbols
        qsgw_symbols = $formula.qsgw_symbols
        upstream_change_ids = $formula.change_ids
        upstream_commits = $commits
        contract_before = $formula.before
        contract_after = $formula.after
        classification = $formula.classification
        required_change = if ($formula.classification -eq 'U2') { 'qsgw_adapter' } else { 'none' }
        benchmark_ids = $formula.benchmark_ids
        evidence = @($formula.evidence)
    }
})
$data.formula_code_impacts = @(
    @($data.formula_code_impacts | Where-Object {
        $_.formula_id -notin $formulaIds -and
        $_.formula_id -notin $retiredFormulaIds
    }) +
    $generatedFormulaRows
)
$data.shared_core_hunks = @($data.shared_core_hunks | Where-Object {
    $_.id -notin $retiredSharedHunkIds
})
$data.approvals = @($data.approvals | Where-Object {
    $_.id -notin $retiredApprovalIds
})

$oldCommit = $data.upstream_inventory.old_commit
$commitHashes = @(Invoke-GitLines @('rev-list', '--reverse', "$oldCommit..$upstream"))
$data.upstream_inventory.commit_hashes = $commitHashes
$retainedCommitMap = @($data.upstream_inventory.commit_change_map |
    Where-Object { $_.commit -notin $refreshCommits } |
    ForEach-Object {
        $remainingIds = @($_.change_ids | Where-Object {
            $_ -notin $retiredChangeIds
        })
        if ($remainingIds.Count -gt 0) {
            [pscustomobject][ordered]@{
                commit = $_.commit
                change_ids = $remainingIds
            }
        }
    })
$data.upstream_inventory.commit_change_map = @(
    $retainedCommitMap +
    @($refreshSpecs | ForEach-Object {
        [pscustomobject][ordered]@{
            commit = $_.commit
            change_ids = @($_.id)
        }
    })
)
$data.upstream_inventory.classified_change_ids = @(
    $data.upstream_changes | ForEach-Object { $_.id }
)
$data.upstream_inventory.coverage_assertion = 'complete'

$gitEvidenceDir = Join-Path $Repository 'qsgw-rebase-evidence\git'
$commitListRelative = 'qsgw-rebase-evidence/git/upstream-commit-list-to-42d3863c.txt'
$nameStatusRelative = 'qsgw-rebase-evidence/git/upstream-name-status-to-42d3863c.txt'
$hunkInventoryRelative = 'qsgw-rebase-evidence/git/upstream-hunk-inventory-to-42d3863c.json'
$commitListPath = Join-Path $gitEvidenceDir 'upstream-commit-list-to-42d3863c.txt'
$nameStatusPath = Join-Path $gitEvidenceDir 'upstream-name-status-to-42d3863c.txt'
$hunkInventoryPath = Join-Path $gitEvidenceDir 'upstream-hunk-inventory-to-42d3863c.json'

[System.IO.File]::WriteAllText(
    $commitListPath,
    ($commitHashes -join [Environment]::NewLine) + [Environment]::NewLine,
    $utf8
)
$nameStatus = @(Invoke-GitLines @('diff', '--name-status', "$oldCommit..$upstream"))
[System.IO.File]::WriteAllText(
    $nameStatusPath,
    ($nameStatus -join [Environment]::NewLine) + [Environment]::NewLine,
    $utf8
)

$baseHunkInventoryPath = Join-Path $gitEvidenceDir 'upstream-hunk-inventory.json'
$hunkInventory = Get-Content -Raw -Encoding UTF8 -LiteralPath $baseHunkInventoryPath |
    ConvertFrom-Json
$hunkInventory.upstream_new = $upstream
$hunkInventory.scope_note =
    'Complete frozen b484f2a9..42d3863c semantic inventory. QSGW inherits upstream shared GW/EXX/symmetry numerics; current protected shared diff is zero.'
$retainedHunkCommits = @($hunkInventory.commits |
    Where-Object { $_.commit -notin $refreshCommits } |
    ForEach-Object {
        $remainingIds = @($_.change_ids | Where-Object {
            $_ -notin $retiredChangeIds
        })
        if ($remainingIds.Count -gt 0) {
            [pscustomobject][ordered]@{
                commit = $_.commit
                change_ids = $remainingIds
            }
        }
    })
$hunkInventory.commits = @(
    $retainedHunkCommits +
    @($refreshSpecs | ForEach-Object {
        [pscustomobject][ordered]@{
            commit = $_.commit
            change_ids = @($_.id)
        }
    })
)
$generatedHunkChanges = @($refreshSpecs | ForEach-Object {
    $spec = $_
    $resolution = switch ($spec.classification) {
        'U0' { 'accept upstream unchanged and retain the build/upstream regression observer' }
        'U1' { 'accept upstream unchanged and observe upstream G0W0 plus QSGW iteration zero' }
        'U2' { 'accept upstream and validate the contract only through QSGW-owned adapters' }
    }
    [pscustomobject][ordered]@{
        id = $spec.id
        commit = $spec.commit
        classification = $spec.classification
        paths = @(Get-CommitPaths $spec.commit)
        symbols = $spec.symbols
        semantic_impact = New-SemanticAudit `
            -Classification $spec.classification `
            -ChangedFields $spec.semantic_changes `
            -Evidence $spec.evidence
        qsgw_reachability = $spec.summary
        resolution = $resolution
    }
})
$hunkInventory.changes = @(
    @($hunkInventory.changes | Where-Object {
        $_.id -notin $refreshIds -and $_.id -notin $retiredChangeIds
    }) +
    $generatedHunkChanges
)
[System.IO.File]::WriteAllText(
    $hunkInventoryPath,
    ($hunkInventory | ConvertTo-Json -Depth 100) + [Environment]::NewLine,
    $utf8
)

Set-HashedArtifact `
    -Record $data.upstream_inventory.commit_list `
    -RelativePath $commitListRelative `
    -Reference 'git rev-list --reverse upstream_old..upstream_new'
Set-HashedArtifact `
    -Record $data.upstream_inventory.name_status `
    -RelativePath $nameStatusRelative `
    -Reference 'git diff --name-status upstream_old..upstream_new'
Set-HashedArtifact `
    -Record $data.upstream_inventory.hunk_inventory `
    -RelativePath $hunkInventoryRelative `
    -Reference 'reviewed semantic hunk inventory with exact commit/change coverage'

$data.current_gate = if ($gate0) {
    'g0w0-upstream-vs-rebased'
} else {
    'build-unit-upstream-regressions'
}
$verified = @($data.verified_evidence) + @(
    'qsgw-rebase-evidence/impact/revised-goal-audit-20260720.md',
    'qsgw-rebase-evidence/validation/local-audit-20260721.md',
    'qsgw-rebase-evidence/validation/generate-layered-staging-plan.ps1',
    'qsgw-rebase-evidence/validation/verify-layered-staging-plan.ps1',
    'qsgw-rebase-evidence/validation/layered-staging-plan-20260722.json',
    'qsgw-rebase-evidence/git/upstream-master-compare-20260722.json',
    'qsgw-rebase-evidence/remote/fish-reader-binding-20260720/v2-postcheck-v1',
    'qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720',
    'qsgw-rebase-evidence/remote/fish-gate-d-current-20260721'
)
$data.verified_evidence = @($verified | Select-Object -Unique)
if ($gate0) {
    $data.verified_evidence = @(
        @($data.verified_evidence) +
        'qsgw-rebase-evidence/remote/fish-gate0-current-20260721/66bfe1cf-v1'
    ) | Select-Object -Unique
}

$openIssues = @(
    [pscustomobject]@{
        id = 'ISSUE-REMOTE-GATE0'
        class_or_gate = 'build-unit-upstream-regressions'
        blocker = 'The current source has no immutable fish build or complete 63-test CTest result.'
        required = 'Build the clean candidate on fish and pass exactly 63 tests with zero failed and zero Not Run.'
    },
    [pscustomobject]@{
        id = 'ISSUE-SYMMETRY-ORACLE'
        class_or_gate = 'abacus-symmetry-two-round'
        blocker = 'Historical capability evidence is not bound to the new clean candidate, and the compatibility harness is not raw legacy source.'
        required = 'Run same-bundle no-mix miniter2 and linear-beta-0.2 miniter5 comparisons with explicit raw-source versus harness provenance.'
    },
    [pscustomobject]@{
        id = 'ISSUE-HARTREE-E2E'
        class_or_gate = 'hartree-no-symmetry-and-symmetry'
        blocker = 'Hartree unit and formula tests pass, but no accepted clean-candidate two-round end-to-end gate exists.'
        required = 'Validate live density, charge, units, q=0 convention, Hermiticity, and legacy equivalence.'
    },
    [pscustomobject]@{
        id = 'ISSUE-QSGW-BAND-E2E'
        class_or_gate = 'qsgw-band-and-h-cut'
        blocker = 'Operator Fourier, band output, CSR, cut, and current linear post-mix closure tests pass, but D0/D1 numerical artifacts are not accepted and no same-dataset legacy/current linear cut-ordering result exists.'
        required = 'Compare uncut and cut modes 0/1/2 for H(k), H(R)/CSR, band eigenvalues, gaps, and PyATB export, including a controlled linear-mixing legacy/current run.'
    },
    [pscustomobject]@{
        id = 'ISSUE-REGRESSION-PLACEHOLDERS'
        class_or_gate = 'formal-regression'
        blocker = 'QSGW testsuite.xml entries are disabled historical placeholders with no committed oracle-backed dataset.'
        required = 'Commit a small two-round ABACUS dataset and references before enabling formal cases.'
    }
)
if ($gate0) {
    $openIssues = @(
        $openIssues | Where-Object { $_.id -ne 'ISSUE-REMOTE-GATE0' }
    )
}
if (-not $hasCleanCandidate) {
    $cleanCandidateIssue = [pscustomobject]@{
        id = 'ISSUE-CLEAN-CANDIDATE'
        class_or_gate = 'layered-commits'
        blocker = 'The current audited source is still an uncommitted working tree; the approved Git index write was rejected by the platform approval quota until 2026-07-25 15:02.'
        required = 'Create the approved local commit layers and bind all later evidence to their exact hashes.'
    }
    $openIssues = @($cleanCandidateIssue) + @($openIssues)
}
$data.open_issues = @($openIssues)

if ($gate0) {
    $data.next_actions = @(
        [pscustomobject]@{
            id = 'next-g0w0-direct-ab'
            action = 'Run byte-identical-input upstream versus candidate G0W0 direct A/B on fish.'
            gate = 'g0w0-upstream-vs-rebased'
            owner = 'rebase operator'
            status = 'ready'
        }
    )
}
elseif ($hasCleanCandidate) {
    $data.next_actions = @(
        [pscustomobject]@{
            id = 'next-clean-candidate-gate0'
            action = 'Run the 63-test fish Gate 0 for the immutable candidate when network access returns.'
            gate = 'build-unit-upstream-regressions'
            owner = 'rebase operator'
            status = 'ready'
        }
    )
}
else {
    $data.next_actions = @(
        [pscustomobject]@{
            id = 'next-layered-candidate-and-gate0'
            action = 'Create the approved local commit layers when Git index approval returns, then run the 63-test fish Gate 0 when network access returns.'
            gate = 'build-unit-upstream-regressions'
            owner = 'rebase operator'
            status = 'ready'
        }
    )
}

$data.planning_state.frozen_dirty_source = -not $hasCleanCandidate
$data.planning_state.clean_candidate_available = $hasCleanCandidate
$data.planning_state.u3_head_matrix_getter =
    'removed_shared_diff_zero'
$data.planning_state.crystal_symmetry_qsgw =
    'candidate_symmetry_parity_supporting_clean_candidate_iterative_gate_pending'
Set-Property -Object $data.planning_state -Name 'qsgw_iterative_headwing' -Value 'unsupported_fail_fast_verified'
Set-Property -Object $data.planning_state -Name 'qsgw_band_operator_fourier' -Value 'wired_unit_and_linear_postmix_observer_verified_remote_gate_pending'
Set-Property -Object $data.planning_state -Name 'hartree' -Value 'unit_verified_remote_gate_pending'
Set-Property -Object $data.planning_state -Name 'expected_ctest_count' -Value 63
Set-Property -Object $data.planning_state -Name 'fish_gate0' -Value $(if ($gate0) {
    'accepted_39_upstream_63_candidate_10_focused_protected_diff_empty'
} else {
    'pending'
})
Set-Property -Object $data.planning_state -Name 'formal_qsgw_regression' -Value 'disabled_historical_placeholders_no_committed_cases'
Set-Property -Object $data.planning_state -Name 'upstream_master_live_compare' -Value '42d3863c_identical_2026-07-21T18:37:20Z'
Set-Property -Object $data.planning_state -Name 'layered_staging_plan' -Value '388_candidate_137_excluded_zero_unclassified_five_layer_dry_run_passed'

Set-Property $data 'revised_scope' ([pscustomobject]@{
    effective_date = '2026-07-21'
    freeze_parent = $frozenParent
    candidate_source = $candidateSource
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
    audit = 'qsgw-rebase-evidence/validation/local-audit-20260721.md'
})

[System.IO.File]::WriteAllText(
    $Manifest,
    ($data | ConvertTo-Json -Depth 100) + [Environment]::NewLine,
    $utf8
)
