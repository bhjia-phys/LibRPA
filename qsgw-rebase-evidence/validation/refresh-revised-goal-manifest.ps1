param(
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Manifest = (Join-Path $Repository 'qsgw-rebase-manifest.json'),
    [string]$CandidateSourceCommit = '',
    [string]$Gate0Evidence = '',
    [string]$Gate1Evidence = '',
    [string]$Gate2Evidence = ''
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$upstream = '67b9888dac0d09870361398165d0b3c1acc931ff'
$frozenParent = $upstream
$noSymCandidate = 'c27482016f70ece5a0e5ccad7199d93ac3f6ebf5'
$branch = 'codex/qsgw-symmetry-no-headwing-67b-20260723'

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
    param(
        [string]$Root,
        [string]$ChecksumFile,
        [string[]]$AllowedMissing = @()
    )

    $rootFull = [IO.Path]::GetFullPath($Root)
    $missing = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal
    )
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
            throw "Checksum path escapes evidence root: $relative"
        }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            if ($relative -in $AllowedMissing) {
                $null = $missing.Add($relative)
                continue
            }
            throw "Evidence checksum target is missing: $relative"
        }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "Evidence checksum mismatch for ${relative}: $actual"
        }
    }
    foreach ($relative in $AllowedMissing) {
        if (-not $missing.Contains($relative)) {
            throw "Expected archive-only checksum target was not missing: $relative"
        }
    }
}

function Assert-TarMemberHash {
    param(
        [string]$Archive,
        [string]$Member,
        [string]$ExpectedSha256
    )

    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) (
        'librpa-gate2-' + [guid]::NewGuid().ToString('N')
    )
    $null = New-Item -ItemType Directory -Path $tempRoot
    try {
        & tar -xzf $Archive -C $tempRoot ("./" + $Member)
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to extract $Member from Gate 2 archive"
        }
        $extracted = Join-Path $tempRoot ($Member.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $extracted -PathType Leaf)) {
            throw "Gate 2 archive member is missing after extraction: $Member"
        }
        $actual = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $extracted
        ).Hash.ToLowerInvariant()
        if ($actual -ne $ExpectedSha256) {
            throw "Gate 2 archive member checksum mismatch for ${Member}: $actual"
        }
    }
    finally {
        if (Test-Path -LiteralPath $tempRoot -PathType Container) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force
        }
    }
}

function Get-LfTextSha256 {
    param([string]$Path)

    $text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $Path))
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(
        $text.Replace("`r`n", "`n")
    )
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return (($sha.ComputeHash($bytes) | ForEach-Object {
            $_.ToString('x2')
        }) -join '')
    }
    finally {
        $sha.Dispose()
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
        'qsgw-rebase-evidence\remote\fish-gate0-current-20260723\4f9ab0cf-v1'
    )
}
$gate0 = $null
$gate0Reference =
    'qsgw-rebase-evidence/remote/fish-gate0-current-20260723/4f9ab0cf-v1/PROVENANCE.txt'
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
        gate = 'fish_gate0_current_v2'
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
        'qsgw-rebase-evidence\remote\fish-gate0-current-20260723\run_fish_gate0_current_v2.sh'
    )
    $runnerHash = Get-LfTextSha256 -Path $runnerPath
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

if ([string]::IsNullOrWhiteSpace($Gate1Evidence)) {
    $Gate1Evidence = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate1-current-20260723\36d74369-recovery-v1'
    )
}
$gate1 = $null
$gate1Reference =
    'qsgw-rebase-evidence/remote/fish-gate1-current-20260723/36d74369-recovery-v1/PROVENANCE.txt'
if (Test-Path -LiteralPath $Gate1Evidence -PathType Container) {
    $green = Join-Path $Gate1Evidence 'GREEN_CONFIRMED'
    $failed = Join-Path $Gate1Evidence 'FAILED'
    $provenancePath = Join-Path $Gate1Evidence 'PROVENANCE.txt'
    $checksumPath = Join-Path $Gate1Evidence 'OUTPUT_SHA256SUMS.txt'
    $sourceManifestPath = Join-Path $Gate1Evidence 'SOURCE_RUN_SHA256SUMS.txt'
    $sourceArchivePath = Join-Path $Gate1Evidence 'SOURCE_ARCHIVE_SHA256SUMS.txt'
    if (-not (Test-Path -LiteralPath $green -PathType Leaf) -or
        (Test-Path -LiteralPath $failed) -or
        -not (Test-Path -LiteralPath $provenancePath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $checksumPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $sourceManifestPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $sourceArchivePath -PathType Leaf)) {
        throw 'Gate 1 evidence is incomplete or contains FAILED'
    }
    Assert-ChecksumManifest -Root $Gate1Evidence -ChecksumFile $checksumPath
    Assert-ChecksumManifest -Root $Gate1Evidence -ChecksumFile $sourceArchivePath
    $sourceManifestRows = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal
    )
    foreach ($line in Get-Content -LiteralPath $sourceManifestPath) {
        $null = $sourceManifestRows.Add($line.Replace('  ./', '  source-run/'))
    }
    foreach ($line in Get-Content -LiteralPath $sourceArchivePath) {
        if (-not $sourceManifestRows.Contains($line)) {
            throw "Gate 1 source archive is not bound by full source manifest: $line"
        }
    }
    $gate1 = Read-KeyValueFile -Path $provenancePath
    $expectedGate1 = @{
        gate = 'fish_gate1_current_g0w0_ab_recovery_v2'
        acceptance = 'true'
        source_run_status = 'rejected_observer_threshold_only'
        upstream_commit = $upstream
        candidate_commit = $candidateSource
        sigc_block_count = '48'
        energy_qp_state_count = '2816'
        symmetry = 'exx_on_gw_on_rpa_on'
        headwing = 'off'
        hartree = 'off'
        band = 'off'
    }
    foreach ($key in $expectedGate1.Keys) {
        if ($gate1[$key] -ne $expectedGate1[$key]) {
            throw "Gate 1 provenance mismatch for ${key}: $($gate1[$key])"
        }
    }
    $runnerPath = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate1-current-20260723\recover_fish_gate1_current_v2.sh'
    )
    $runnerHash = Get-LfTextSha256 -Path $runnerPath
    if ($runnerHash -ne $gate1.recovery_runner_sha256) {
        throw "Gate 1 recovery runner hash mismatch: $runnerHash"
    }
    $sourceManifestHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifestPath
    ).Hash.ToLowerInvariant()
    if ($sourceManifestHash -ne $gate1.source_run_manifest_sha256) {
        throw "Gate 1 source manifest hash mismatch: $sourceManifestHash"
    }
}

if ([string]::IsNullOrWhiteSpace($Gate2Evidence)) {
    $Gate2Evidence = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate2-current-20260723\dd7a75f2-v1'
    )
}
$gate2 = $null
$gate2Reference =
    'qsgw-rebase-evidence/remote/fish-gate2-current-20260723/dd7a75f2-v1/PROVENANCE.txt'
if (Test-Path -LiteralPath $Gate2Evidence -PathType Container) {
    $green = Join-Path $Gate2Evidence 'GREEN_CONFIRMED'
    $failed = Join-Path $Gate2Evidence 'FAILED'
    $provenancePath = Join-Path $Gate2Evidence 'PROVENANCE.txt'
    $checksumPath = Join-Path $Gate2Evidence 'OUTPUT_SHA256SUMS.txt'
    $archiveManifestPath = Join-Path $Gate2Evidence 'ARCHIVE_SHA256SUMS.txt'
    $archivePath = Join-Path $Gate2Evidence (
        'librpa-qsgw-gate2-current-20260723-dd7a75f2-v1-files.tar.gz'
    )
    if (-not (Test-Path -LiteralPath $green -PathType Leaf) -or
        (Test-Path -LiteralPath $failed) -or
        -not (Test-Path -LiteralPath $provenancePath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $checksumPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $archiveManifestPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw 'Gate 2 evidence is incomplete or contains FAILED'
    }
    Assert-ChecksumManifest -Root $Gate2Evidence `
        -ChecksumFile $archiveManifestPath
    Assert-ChecksumManifest -Root $Gate2Evidence `
        -ChecksumFile $checksumPath `
        -AllowedMissing @('candidate-qsgw/qsgw_matrices.dat')
    Assert-TarMemberHash -Archive $archivePath `
        -Member 'candidate-qsgw/qsgw_matrices.dat' `
        -ExpectedSha256 '64cf0e2712697f7cabc4350aa9b28ac6cda9d1a51a5a20189dc3db2ed708b382'

    $gate2 = Read-KeyValueFile -Path $provenancePath
    $expectedGate2 = @{
        gate = 'fish_gate2_current_qsgw_first_self_energy_v2'
        acceptance = 'true'
        runner_commit = 'dd7a75f22cb4ce114a82bc720fe11f60377044ae'
        upstream_commit = $upstream
        candidate_commit = $candidateSource
        candidate_executable_sha256 = '77ab964e15f0cdfee05ad54cf5b9da4ad9e4e3ac1f6990c4b50e8f0a55a29b47'
        gate0_provenance_sha256 = '62a4026017c26b28f9c9503fb4c71a265345369db2a709b0f811a7000b5fd424'
        gate1_accepted_provenance_sha256 = '880115845b7cb5a983885613a1b2325607bb269836f681b0af5051b8f9d74d0d'
        gate1_accepted_output_manifest_sha256 = '74a3453c46df34e7b0ec8430260cd47182e6d3d01a409f4a313dc976948a127f'
        gate1_source_manifest_sha256 = '22840e6cf84d18526f05283231c47eb0a1784c58764f4387a7391c1f50aa5155'
        qsgw_iter1_invariants_sha256 = '116035f4a52a81fa2ed69bd2906958ea67435a829ab7dd49e770997c5115b534'
        qsgw_iter1_g0w0_sigc_comparison_sha256 = 'f117c50decde276181c3fa9a1b44c27026a63b6e80754d8ebf99a30f7a97faf8'
        semantic_iteration_zero = 'immutable_initial_state'
        semantic_first_self_energy = 'trace_iteration_1_channel_0'
        sigc_block_count = '48'
        symmetry = 'exx_on_gw_on_rpa_on'
        headwing = 'off'
        hartree = 'off'
        band = 'off'
        mixing = 'none'
        iterations = '0:1'
    }
    foreach ($key in $expectedGate2.Keys) {
        if ($gate2[$key] -ne $expectedGate2[$key]) {
            throw "Gate 2 provenance mismatch for ${key}: $($gate2[$key])"
        }
    }
    $runnerPath = Join-Path $Repository (
        'qsgw-rebase-evidence\remote\fish-gate2-current-20260723\run_fish_gate2_current_v2.sh'
    )
    $runnerHash = Get-LfTextSha256 -Path $runnerPath
    if ($runnerHash -ne $gate2.runner_sha256) {
        throw "Gate 2 runner hash mismatch: $runnerHash"
    }

    $comparisonPath = Join-Path $Gate2Evidence (
        'qsgw-iter1-vs-upstream-g0w0-sigc.json'
    )
    $comparison = Get-Content -Raw -Encoding UTF8 -LiteralPath $comparisonPath |
        ConvertFrom-Json
    if (-not $comparison.passed -or
        $comparison.block_count -ne 48 -or
        $comparison.iteration -ne 1 -or
        $comparison.semantic_iteration_zero -ne 'immutable_initial_state' -or
        $comparison.semantic_first_self_energy -ne 'trace_iteration_1_channel_0' -or
        $comparison.max_abs_difference_ha -gt 1e-10 -or
        $comparison.relative_frobenius_difference -gt 1e-10) {
        throw 'Gate 2 QSGW versus G0W0 comparison did not satisfy acceptance'
    }
    $invariantsPath = Join-Path $Gate2Evidence 'qsgw-iter1-invariants.json'
    $invariants = Get-Content -Raw -Encoding UTF8 -LiteralPath $invariantsPath |
        ConvertFrom-Json
    if (-not $invariants.passed -or
        $invariants.none_mixer_max_abs_ha -ne 0 -or
        $invariants.fixed_basis_wfc_rotation_relative_frobenius -ne 0 -or
        $invariants.raw_h_closure_max_abs_ha -gt 1e-10 -or
        $invariants.hermiticity_max_abs_ha -gt 1e-10) {
        throw 'Gate 2 fixed-basis or Hamiltonian invariants did not satisfy acceptance'
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
    'qsgw-rebase-evidence/git/upstream-refresh-42d3863c-to-67b9888d.md'

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
            'qsgw-rebase-evidence/remote/fish-gate0-current-20260723/4f9ab0cf-v1/' +
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
if ($gate1) {
    $gate1Benchmark = @(
        $data.benchmarks | Where-Object { $_.id -eq 'g0w0-upstream-vs-rebased' }
    )[0]
    $artifactSpecs = @(
        @('upstream-g0w0-log', 'source-run/upstream/librpa.stdout'),
        @('rebased-g0w0-log', 'source-run/candidate/librpa.stdout'),
        @('g0w0-comparison', 'g0w0-comparison.json'),
        @('energy-qp-comparison', 'energy-qp-comparison.json'),
        @('gate1-provenance', 'PROVENANCE.txt'),
        @('gate1-output-manifest', 'OUTPUT_SHA256SUMS.txt'),
        @('gate1-source-manifest', 'SOURCE_RUN_SHA256SUMS.txt'),
        @('gate1-source-archive-manifest', 'SOURCE_ARCHIVE_SHA256SUMS.txt')
    )
    $gate1Artifacts = @()
    foreach ($spec in $artifactSpecs) {
        $relative =
            'qsgw-rebase-evidence/remote/fish-gate1-current-20260723/36d74369-recovery-v1/' +
            $spec[1]
        $record = [pscustomobject][ordered]@{
            id = $spec[0]
            path = $null
            sha256 = $null
            reference = $null
        }
        Set-HashedArtifact -Record $record -RelativePath $relative `
            -Reference $gate1Reference
        $gate1Artifacts += $record
    }
    $gate1Benchmark.status = 'accepted'
    $gate1Benchmark.artifacts = @($gate1Artifacts)
}
if ($gate2) {
    $gate2Benchmark = @(
        $data.benchmarks | Where-Object { $_.id -eq 'qsgw-iter0-vs-upstream' }
    )[0]
    $artifactSpecs = @(
        @('upstream-g0w0-tensor', 'gate1-SOURCE_RUN_SHA256SUMS.txt'),
        @('qsgw-iter0-tensor', 'librpa-qsgw-gate2-current-20260723-dd7a75f2-v1-files.tar.gz'),
        @('iteration-zero-comparison', 'qsgw-iter1-vs-upstream-g0w0-sigc.json'),
        @('adapter-contract-comparison', 'qsgw-iter1-invariants.json'),
        @('gate2-provenance', 'PROVENANCE.txt'),
        @('gate2-output-manifest', 'OUTPUT_SHA256SUMS.txt'),
        @('gate2-archive-manifest', 'ARCHIVE_SHA256SUMS.txt')
    )
    $gate2Artifacts = @()
    foreach ($spec in $artifactSpecs) {
        $relative =
            'qsgw-rebase-evidence/remote/fish-gate2-current-20260723/dd7a75f2-v1/' +
            $spec[1]
        $record = [pscustomobject][ordered]@{
            id = $spec[0]
            path = $null
            sha256 = $null
            reference = $null
        }
        Set-HashedArtifact -Record $record -RelativePath $relative `
            -Reference $gate2Reference
        $gate2Artifacts += $record
    }
    $gate2Benchmark.status = 'accepted'
    $gate2Benchmark.artifacts = @($gate2Artifacts)
}
$data.upstream_inventory.new_commit = $upstream

$refreshAuditEarly =
    'qsgw-rebase-evidence/git/upstream-refresh-1376ee4f-to-95c4c080.md'
$refreshAuditLate =
    'qsgw-rebase-evidence/git/upstream-refresh-95c4c080-to-42d3863c.md'
$refreshAuditCurrent =
    'qsgw-rebase-evidence/git/upstream-refresh-42d3863c-to-67b9888d.md'
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
    },
    [pscustomobject]@{
        id = 'UP-ELPA-DEVICE-ALLOC-67B-001'
        commit = '5e390487218a834e645b915f47006c969f440f0c'
        classification = 'U1'
        summary = 'Adopts the current LibDDLA device allocation and free API in the ELPA connector.'
        semantic_changes = @('api', 'call_ordering')
        symbols = @('deviceMalloc', 'deviceFree')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-DDLA-BUNDLE-67B-001'
        commit = 'e99cdf016f825dd531203512795ffe25d9670ea8'
        classification = 'U1'
        summary = 'Updates the bundled LibDDLA API, solvers, transport, device memory, and build integration.'
        semantic_changes = @('api', 'matrix_layout', 'mpi_ownership', 'call_ordering')
        symbols = @('LibDDLA', 'deviceMalloc', 'deviceFree')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-HEADWING-BODY-SOLVE-67B-001'
        commit = '85968a20b960f8d8be115ce72dcef312a45ac27c'
        classification = 'U1'
        summary = 'Forms the head-wing body inverse by solving B X = I with the upstream distributed solver.'
        semantic_changes = @('call_ordering')
        symbols = @('invert_headwing_body_with_identity_solve', 'diele_func::get_body_inv')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-HEAD-RANK1-67B-001'
        commit = '054df5b78034e2f9878f4348738ee09e3b59186c'
        classification = 'U1'
        summary = 'Applies the Gamma head correction as a rank-one update without rotating the full dielectric matrix.'
        semantic_changes = @('basis', 'call_ordering')
        symbols = @('diele_func::rewrite_eps')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-DDLA-REVISION-67B-001'
        commit = '0b9bdedb2bdf5b60d469aa3bb8351ffb93b2ecfb'
        classification = 'U0'
        summary = 'Updates only the top-level bundled LibDDLA revision metadata.'
        semantic_changes = @()
        symbols = @()
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-HEADWING-BODY-CLEANUP-67B-001'
        commit = 'bcf3e573bedce6b742bb70b4594b9db4f8ebfca0'
        classification = 'U1'
        summary = 'Removes redundant head-wing body-inverse allocation and setup.'
        semantic_changes = @('call_ordering')
        symbols = @('diele_func::get_body_inv', 'diele_func::rewrite_eps')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        id = 'UP-DIELECTRIC-SOLVE-ERROR-67B-001'
        commit = '67b9888dac0d09870361398165d0b3c1acc931ff'
        classification = 'U1'
        summary = 'Reports a nonzero dielectric solve status as a runtime failure.'
        semantic_changes = @('call_ordering')
        symbols = @('invert_headwing_body_with_identity_solve')
        evidence = $refreshAuditCurrent
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
    },
    [pscustomobject]@{
        formula_id = 'F-DDLA-DEVICE-CONTRACT-67B'
        formula = 'A distributed device allocation, transfer, factorization, and solve preserves the matrix represented by its DDLA handle.'
        upstream_symbols = @('LibDDLA', 'deviceMalloc', 'deviceFree')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-ELPA-DEVICE-ALLOC-67B-001', 'UP-DDLA-BUNDLE-67B-001')
        classification = 'U1'
        before = New-FormulaContract 'distributed solver basis' 'pre-67b DDLA buffers and API' 'matrix-native units' 'DDLA handle and MPI ownership' 'unchanged'
        after = New-FormulaContract 'distributed solver basis' 'current DDLA buffers and API' 'matrix-native units' 'DDLA handle and MPI ownership' 'unchanged'
        benchmark_ids = @('build-upstream-regressions', 'g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        formula_id = 'F-HEADWING-BODY-INVERSE-67B'
        formula = 'The head-wing body inverse X is obtained from B X = I in the unchanged dielectric body basis.'
        upstream_symbols = @('invert_headwing_body_with_identity_solve', 'diele_func::get_body_inv', 'diele_func::rewrite_eps')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-HEADWING-BODY-SOLVE-67B-001', 'UP-HEADWING-BODY-CLEANUP-67B-001', 'UP-DIELECTRIC-SOLVE-ERROR-67B-001')
        classification = 'U1'
        before = New-FormulaContract 'dielectric body basis' 'explicit body inverse path' 'dimensionless' 'upstream distributed matrix ownership' 'unchanged'
        after = New-FormulaContract 'dielectric body basis' 'identity solve with explicit failure reporting' 'dimensionless' 'upstream distributed matrix ownership' 'unchanged'
        benchmark_ids = @('build-upstream-regressions', 'g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditCurrent
    },
    [pscustomobject]@{
        formula_id = 'F-GAMMA-HEAD-RANK1-67B'
        formula = 'epsilon is corrected by (H - x1^H epsilon x1) x1 x1^H in the original Coulomb representation.'
        upstream_symbols = @('diele_func::rewrite_eps')
        qsgw_symbols = @('run_qsgw')
        change_ids = @('UP-HEAD-RANK1-67B-001')
        classification = 'U1'
        before = New-FormulaContract 'rotated Coulomb basis' 'full rotate-correct-unrotate path' 'dimensionless' 'upstream dielectric ownership' 'unchanged'
        after = New-FormulaContract 'original Coulomb representation' 'rank-one head correction' 'dimensionless' 'upstream dielectric ownership' 'unchanged'
        benchmark_ids = @('build-upstream-regressions', 'g0w0-upstream-vs-rebased', 'qsgw-iter0-vs-upstream')
        evidence = $refreshAuditCurrent
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
$commitListRelative = 'qsgw-rebase-evidence/git/upstream-commit-list-to-67b9888d.txt'
$nameStatusRelative = 'qsgw-rebase-evidence/git/upstream-name-status-to-67b9888d.txt'
$hunkInventoryRelative = 'qsgw-rebase-evidence/git/upstream-hunk-inventory-to-67b9888d.json'
$commitListPath = Join-Path $gitEvidenceDir 'upstream-commit-list-to-67b9888d.txt'
$nameStatusPath = Join-Path $gitEvidenceDir 'upstream-name-status-to-67b9888d.txt'
$hunkInventoryPath = Join-Path $gitEvidenceDir 'upstream-hunk-inventory-to-67b9888d.json'

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
    'Complete frozen b484f2a9..67b9888d semantic inventory. QSGW inherits upstream shared GW/EXX/symmetry numerics; current protected shared diff is zero.'
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

$data.current_gate = if ($gate2) {
    'solid-qsgw-no-mixing-old-vs-new'
} elseif ($gate1) {
    'qsgw-iter0-vs-upstream-g0w0'
} elseif ($gate0) {
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
    'qsgw-rebase-evidence/git/upstream-master-ls-remote-20260723.json',
    'qsgw-rebase-evidence/git/upstream-refresh-42d3863c-to-67b9888d.md',
    'qsgw-rebase-evidence/remote/fish-reader-binding-20260720/v2-postcheck-v1',
    'qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720',
    'qsgw-rebase-evidence/remote/fish-gate-d-current-20260721'
)
$data.verified_evidence = @($verified | Select-Object -Unique)
if ($gate0) {
    $data.verified_evidence = @(
        @($data.verified_evidence) +
        'qsgw-rebase-evidence/remote/fish-gate0-current-20260723/4f9ab0cf-v1'
    ) | Select-Object -Unique
}
if ($gate1) {
    $data.verified_evidence = @(
        @($data.verified_evidence) +
        'qsgw-rebase-evidence/remote/fish-gate1-current-20260723/36d74369-recovery-v1'
    ) | Select-Object -Unique
}
if ($gate2) {
    $data.verified_evidence = @(
        @($data.verified_evidence) +
        'qsgw-rebase-evidence/remote/fish-gate2-current-20260723/dd7a75f2-v1'
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
        id = 'ISSUE-REMOTE-GATE1'
        class_or_gate = 'g0w0-upstream-vs-rebased'
        blocker = 'The current upstream/candidate G0W0 comparison has no accepted immutable fish evidence.'
        required = 'Run and archive the byte-identical-input upstream versus candidate G0W0 comparison.'
    },
    [pscustomobject]@{
        id = 'ISSUE-REMOTE-GATE2'
        class_or_gate = 'qsgw-iter0-vs-upstream-g0w0'
        blocker = 'The current QSGW first self-energy has no accepted immutable comparison against upstream G0W0.'
        required = 'Run and archive the fixed-basis first-self-energy comparison and structural invariants.'
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
if ($gate1) {
    $openIssues = @(
        $openIssues | Where-Object { $_.id -ne 'ISSUE-REMOTE-GATE1' }
    )
}
if ($gate2) {
    $openIssues = @(
        $openIssues | Where-Object { $_.id -ne 'ISSUE-REMOTE-GATE2' }
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

if ($gate2) {
    $data.next_actions = @(
        [pscustomobject]@{
            id = 'next-formal-a1'
            action = 'Bind accepted Gate 2 and run fresh legacy-full-BZ versus current-full-BZ none-miniter2 and linear-beta-0.2-miniter5 comparisons on fish.'
            gate = 'solid-qsgw-no-mixing-old-vs-new'
            owner = 'rebase operator'
            status = 'ready'
        }
    )
}
elseif ($gate1) {
    $data.next_actions = @(
        [pscustomobject]@{
            id = 'next-qsgw-iter0-g0w0'
            action = 'Run current QSGW first-self-energy versus accepted upstream G0W0, then launch formal A1.'
            gate = 'qsgw-iter0-vs-upstream-g0w0'
            owner = 'rebase operator'
            status = 'ready'
        }
    )
}
elseif ($gate0) {
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
Set-Property -Object $data.planning_state -Name 'fish_gate1' -Value $(if ($gate1) {
    'accepted_sigc_48_qp_2816_source_outputs_immutable_postcheck'
} else {
    'pending'
})
Set-Property -Object $data.planning_state -Name 'fish_gate2' -Value $(if ($gate2) {
    'accepted_first_self_energy_48_sigc_fixed_basis_and_closure'
} else {
    'pending'
})
Set-Property -Object $data.planning_state -Name 'formal_qsgw_regression' -Value 'disabled_historical_placeholders_no_committed_cases'
Set-Property -Object $data.planning_state -Name 'upstream_master_live_compare' -Value '67b9888d_identical_2026-07-22T17:46:14Z'
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
