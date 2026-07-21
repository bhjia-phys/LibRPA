[CmdletBinding()]
param(
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [switch]$Check,
    [string[]]$AdditionalPathForTest = @()
)

$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$expectedBranch = 'codex/qsgw-symmetry-no-headwing-42d-20260720'
$expectedHead = 'b7273e13c77d5ea781f192cea3c4201710b6f9fa'
$frozenUpstream = '42d3863c1d865194d382a085851d1e2e8a39764f'
$outputRelative =
    'qsgw-rebase-evidence/validation/layered-staging-plan-20260722.json'
$outputPath = Join-Path $Repository ($outputRelative.Replace('/', '\'))

function Invoke-GitLines {
    param([string[]]$Arguments)

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $lines = @(& git -C $Repository @Arguments 2>$null)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode"
    }
    return @(
        $lines |
            Where-Object { $_ -ne $null -and $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Replace('\', '/') }
    )
}

function Test-CuratedTreePath {
    param(
        [string]$Path,
        [string]$Root,
        [string[]]$AllowedChildren
    )

    if (-not $Path.StartsWith($Root, [StringComparison]::Ordinal)) {
        return $false
    }
    $relative = $Path.Substring($Root.Length)
    if ($relative.Length -eq 0) {
        return $false
    }
    if (-not $relative.Contains('/')) {
        return $true
    }
    foreach ($child in $AllowedChildren) {
        if ($relative.StartsWith($child, [StringComparison]::Ordinal)) {
            return $true
        }
    }
    return $false
}

function Test-NeverStagePath {
    param([string]$Path)

    if ($Path -match '(^|/)(__pycache__|\.pytest_cache)(/|$)' -or
        $Path -match '\.(pyc|obj|exe)$' -or
        $Path -match '(^|/)(tmp[^/]*|patch-work-[^/]*|patch-apply-test-[^/]*)(/|$)' -or
        $Path -match '(^|/)[^/]*-test-scratch(/|$)' -or
        $Path -match '(^|/)band-pipeline-[^/]*-p7q2d8gv(/|$)' -or
        $Path -match '(^|/)legacy-runtime\.strings$' -or
        $Path -match '(^|/)manifest_source_check_fixture(/|$)') {
        return $true
    }
    return $false
}

function Test-BytePreservedEvidencePath {
    param([string]$Path)

    if (-not $Path.StartsWith(
            'qsgw-rebase-evidence/remote/',
            [StringComparison]::Ordinal)) {
        return $false
    }
    return (
        $Path -match '\.(patch|stdout|stderr)$' -or
        $Path -match '/oracle-source-audit-v1/' -or
        $Path -match '/[^/]*CMakeCache\.txt$'
    )
}

function Get-PathCategory {
    param([string]$Path)

    if (Test-NeverStagePath $Path) {
        return 'excluded'
    }

    $layer1Exact = @(
        'docs/user_guide/runtime_parameters.yml',
        'driver/driver.cpp',
        'driver/driver.h',
        'driver/inputfile.cpp',
        'driver/tasks/qsgw.cpp',
        'driver/test/test_qsgw_inputfile.cpp',
        'src/test/CMakeLists.txt'
    )
    if ($layer1Exact -contains $Path -or
        $Path.StartsWith('src/qsgw/', [StringComparison]::Ordinal) -or
        $Path -match '^src/test/test_qsgw_[^/]+\.cpp$') {
        return 'layer1'
    }

    $layer2Exact = @(
        'regression_tests/backend/comparisons/cmp_qsgw.py',
        'regression_tests/backend/comparisons/test_cmp_qsgw.py',
        'regression_tests/backend/test_qsgw_driver_wiring.py',
        'regression_tests/testsuite.xml'
    )
    if ($layer2Exact -contains $Path) {
        return 'layer2'
    }

    $abacusChildren = @(
        'dongfang-may-si-k888-reference/',
        'dongfang-si-k888-job2375481/',
        'fish-source-freeze-v3/',
        'producer-inputs-v1/',
        'producer-inputs-v2/'
    )
    $abacusArguments = @{
        Path = $Path
        Root = 'qsgw-rebase-evidence/remote/abacus-pinned-dd421665-20260720/'
        AllowedChildren = $abacusChildren
    }
    $readerArguments = @{
        Path = $Path
        Root = 'qsgw-rebase-evidence/remote/fish-reader-binding-20260720/'
        AllowedChildren = @('v2-postcheck-v1/')
    }
    $inAbacus = Test-CuratedTreePath @abacusArguments
    $inReaderBinding = Test-CuratedTreePath @readerArguments
    if ($inAbacus -or $inReaderBinding) {
        return 'layer3'
    }

    $layer4SimpleTrees = @(
        'qsgw-rebase-evidence/remote/fish-gate0-current-20260721/',
        'qsgw-rebase-evidence/remote/fish-gate-a-current-20260721/',
        'qsgw-rebase-evidence/remote/fish-gate-c-current-20260721/',
        'qsgw-rebase-evidence/remote/fish-gate-c-legacy-corrected-20260721/',
        'qsgw-rebase-evidence/remote/fish-gate-d-current-20260721/'
    )
    foreach ($root in $layer4SimpleTrees) {
        if (Test-CuratedTreePath $Path $root @()) {
            return 'layer4'
        }
    }
    $symmetryChildren = @(
        'comparator-unit-v1/',
        'historical-legacy-8476213-provenance/',
        'librpa-qsgw-gate-a0-builds-20260720-v3/',
        'observer-tools-v1/',
        'oracle-source-audit-v1/'
    )
    $symmetryArguments = @{
        Path = $Path
        Root = 'qsgw-rebase-evidence/remote/fish-gate-a-symmetry-20260720/'
        AllowedChildren = $symmetryChildren
    }
    $baselineArguments = @{
        Path = $Path
        Root = 'qsgw-rebase-evidence/remote/dongfang-historical-toolchain-baseline-2380797/'
        AllowedChildren = @('runtime-parent/')
    }
    $inSymmetryEvidence = Test-CuratedTreePath @symmetryArguments
    $inHistoricalBaseline = Test-CuratedTreePath @baselineArguments
    if ($inSymmetryEvidence -or $inHistoricalBaseline) {
        return 'layer4'
    }

    $layer5Exact = @(
        'QSGW_REBASE_PLAN.md',
        'qsgw-rebase-manifest.json',
        'qsgw-rebase-evidence/validation/refresh-revised-goal-manifest.ps1',
        'qsgw-rebase-evidence/validation/generate-layered-staging-plan.ps1',
        'qsgw-rebase-evidence/validation/verify-layered-staging-plan.ps1',
        'qsgw-rebase-evidence/validation/layered-staging-plan-20260722.json',
        'qsgw-rebase-evidence/validation/local-audit-20260721.md',
        'qsgw-rebase-evidence/validation/layered-commit-plan-20260721.md',
        'qsgw-rebase-evidence/validation/hartree-legacy-oracle-audit-20260721.md',
        'qsgw-rebase-evidence/validation/manifest-validator-20260721.txt',
        'qsgw-rebase-evidence/validation/manifest-validator-20260721.exit-code.txt',
        'qsgw-rebase-evidence/git/upstream-commit-list-to-42d3863c.txt',
        'qsgw-rebase-evidence/git/upstream-name-status-to-42d3863c.txt',
        'qsgw-rebase-evidence/git/upstream-hunk-inventory-to-42d3863c.json',
        'qsgw-rebase-evidence/git/upstream-master-compare-20260722.json',
        'qsgw-rebase-evidence/git/upstream-refresh-95c4c080-to-42d3863c.md'
    )
    if ($layer5Exact -contains $Path) {
        return 'layer5'
    }
    return 'unclassified'
}

function Get-PathsetSha256 {
    param([string[]]$Records)

    $bytes = [Text.Encoding]::UTF8.GetBytes(($Records -join "`n") + "`n")
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString(
            $algorithm.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

$branch = @(Invoke-GitLines @('branch', '--show-current'))[0]
$head = @(Invoke-GitLines @('rev-parse', 'HEAD'))[0]
if ($branch -ne $expectedBranch) {
    throw "Unexpected branch: $branch"
}
if ($head -ne $expectedHead) {
    throw "Unexpected HEAD: $head"
}
$null = Invoke-GitLines @(
    'merge-base', '--is-ancestor', $frozenUpstream, $head
)

$staged = @(Invoke-GitLines @('diff', '--cached', '--name-only'))
if ($staged.Count -ne 0) {
    throw "Git index must be empty before generating staging lists"
}

$workset = @(
    Invoke-GitLines @(
        'ls-files', '--modified', '--others', '--exclude-standard'
    )
)
$workset += $outputRelative
$workset += @($AdditionalPathForTest | ForEach-Object { $_.Replace('\', '/') })
$workset = @($workset | Sort-Object -Unique)

$categories = [ordered]@{
    layer1 = [Collections.Generic.List[string]]::new()
    layer2 = [Collections.Generic.List[string]]::new()
    layer3 = [Collections.Generic.List[string]]::new()
    layer4 = [Collections.Generic.List[string]]::new()
    layer5 = [Collections.Generic.List[string]]::new()
    excluded = [Collections.Generic.List[string]]::new()
    unclassified = [Collections.Generic.List[string]]::new()
}
foreach ($path in $workset) {
    $category = Get-PathCategory $path
    $categories[$category].Add($path)
}
if ($categories.unclassified.Count -ne 0) {
    throw "Unclassified worktree paths: $($categories.unclassified -join ', ')"
}

$protectedPaths = @(
    'src/core/dielecmodel.cpp', 'src/core/dielecmodel.h',
    'src/core/gw.cpp', 'src/core/gw.h',
    'src/core/exx.cpp', 'src/core/exx.h',
    'src/api/compute_g0w0.cpp', 'src/api/compute_exx.cpp',
    'driver/tasks/g0w0.cpp', 'driver/tasks/g0w0_band.cpp'
)
$protectedArguments = @('diff', '--name-only', $frozenUpstream, '--') +
    $protectedPaths
$protectedDiff = @(Invoke-GitLines $protectedArguments)
if ($protectedDiff.Count -ne 0) {
    throw "Protected G0W0/GW/EXX/API diff is not empty"
}

$layerDefinitions = @(
    [ordered]@{ id = 'layer1'; subject = 'qsgw: complete independent Hartree and band workflows' },
    [ordered]@{ id = 'layer2'; subject = 'test(qsgw): wire formal regression observers' },
    [ordered]@{ id = 'layer3'; subject = 'test(qsgw): freeze pinned ABACUS producer contracts' },
    [ordered]@{ id = 'layer4'; subject = 'test(qsgw): add versioned A C and D gate runners' },
    [ordered]@{ id = 'layer5'; subject = 'docs(qsgw): bind rebase audit to candidate source' }
)
$layers = foreach ($definition in $layerDefinitions) {
    $paths = @($categories[$definition.id] | Sort-Object)
    [ordered]@{
        id = $definition.id
        subject = $definition.subject
        count = $paths.Count
        paths = $paths
    }
}
$records = foreach ($layer in $layers) {
    foreach ($path in $layer.paths) {
        "$($layer.id)|$path"
    }
}
foreach ($path in @($categories.excluded | Sort-Object)) {
    $records += "excluded|$path"
}

$candidatePaths = @(
    foreach ($layer in $layers) {
        foreach ($path in $layer['paths']) {
            $path
        }
    }
)
$candidateCount = $candidatePaths.Count
$distinctCandidateCount = @($candidatePaths | Sort-Object -Unique).Count
if ($distinctCandidateCount -ne $candidateCount) {
    throw "A candidate path occurs in more than one layer"
}
$compareArguments = @{
    ReferenceObject = @($candidatePaths | Sort-Object -Unique)
    DifferenceObject = @($categories.excluded | Sort-Object -Unique)
    IncludeEqual = $true
    ExcludeDifferent = $true
}
$candidateExcludedOverlap = @(Compare-Object @compareArguments)
if ($candidateExcludedOverlap.Count -ne 0) {
    throw "Candidate and excluded path sets overlap"
}
if ($candidateCount + $categories.excluded.Count -ne $workset.Count) {
    throw "Candidate and excluded path sets do not cover the worktree"
}
$whitespaceExemptPaths = @(
    $candidatePaths |
        Where-Object { Test-BytePreservedEvidencePath $_ } |
        Sort-Object -Unique
)
$whitespaceExemptCandidateDifference = @(
    Compare-Object -ReferenceObject $candidatePaths `
        -DifferenceObject $whitespaceExemptPaths |
        Where-Object { $_.SideIndicator -eq '=>' }
)
if ($whitespaceExemptCandidateDifference.Count -ne 0) {
    throw "A whitespace exemption is not a candidate path"
}
$data = [ordered]@{
    schema_version = 1
    repository = [ordered]@{
        branch = $branch
        head = $head
        frozen_upstream = $frozenUpstream
    }
    policy = [ordered]@{
        stage_only_listed_paths = $true
        directory_level_git_add_forbidden = $true
        git_visible_workset_only = $true
        index_must_be_empty = $true
        protected_diff_must_be_empty = $true
    }
    whitespace_check = [ordered]@{
        strict_for_non_exempt_paths = $true
        exemption_reason = 'Byte-preserved remote patches, source snapshots, and tool stdout/stderr/cache evidence must not be rewritten.'
        byte_preserved_evidence_paths = $whitespaceExemptPaths
    }
    counts = [ordered]@{
        workset = $workset.Count
        candidate = $candidateCount
        excluded = $categories.excluded.Count
        unclassified = 0
    }
    pathset_sha256 = Get-PathsetSha256 @($records)
    layers = @($layers)
    excluded_paths = @($categories.excluded | Sort-Object)
}
$serialized = ($data | ConvertTo-Json -Depth 20) + [Environment]::NewLine

if ($Check) {
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "Staging plan does not exist: $outputPath"
    }
    $existing = [IO.File]::ReadAllText($outputPath)
    if ($existing -cne $serialized) {
        throw "Staging plan is stale: $outputPath"
    }
    Write-Output "PASS layered staging plan: $candidateCount candidate, $($categories.excluded.Count) excluded"
}
else {
    [IO.File]::WriteAllText($outputPath, $serialized, $utf8)
    Write-Output "WROTE $outputPath"
    Write-Output "candidate=$candidateCount excluded=$($categories.excluded.Count)"
}
