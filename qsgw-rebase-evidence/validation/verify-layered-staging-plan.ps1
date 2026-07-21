[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('layer1', 'layer2', 'layer3', 'layer4', 'layer5')]
    [string]$Layer,
    [string]$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$Plan = (Join-Path $PSScriptRoot 'layered-staging-plan-20260722.json'),
    [switch]$DryRun,
    [string]$ScratchRoot = ''
)

$ErrorActionPreference = 'Stop'

function Invoke-GitCapture {
    param(
        [string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& git -C $Repository @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($AllowedExitCodes -notcontains $exitCode) {
        $text = @($output | ForEach-Object { $_.ToString() }) -join "`n"
        throw "git $($Arguments -join ' ') failed with exit code $exitCode`n$text"
    }
    return [pscustomobject]@{
        exit_code = $exitCode
        lines = @($output | ForEach-Object { $_.ToString() })
    }
}

function Resolve-GitPath {
    param([string]$Path)

    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $Repository $Path))
}

if (-not (Test-Path -LiteralPath $Plan -PathType Leaf)) {
    throw "Layered staging plan does not exist: $Plan"
}
$data = Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json
if ($data.schema_version -ne 1) {
    throw "Unsupported layered staging plan schema: $($data.schema_version)"
}
$definition = @($data.layers | Where-Object { $_.id -eq $Layer })
if ($definition.Count -ne 1) {
    throw "Layer $Layer is missing or duplicated in the staging plan"
}
$expected = @($definition[0].paths | Sort-Object -Unique)
if ($expected.Count -ne [int]$definition[0].count) {
    throw "Layer $Layer contains duplicate paths"
}
$allExempt = @(
    $data.whitespace_check.byte_preserved_evidence_paths |
        ForEach-Object { $_.ToString() } |
        Sort-Object -Unique
)
$exempt = @(
    $expected |
        Where-Object { $allExempt -contains $_ } |
        Sort-Object -Unique
)
$strict = @(
    $expected |
        Where-Object { $allExempt -notcontains $_ } |
        Sort-Object -Unique
)
foreach ($path in $expected) {
    $absolute = Join-Path $Repository ($path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        throw "Layer $Layer path is not a file: $path"
    }
}

$branch = (Invoke-GitCapture @('branch', '--show-current')).lines[0]
if ($branch -ne $data.repository.branch) {
    throw "Unexpected branch: $branch"
}
$head = (Invoke-GitCapture @('rev-parse', 'HEAD')).lines[0]
$null = Invoke-GitCapture @(
    'merge-base', '--is-ancestor', $data.repository.head, $head
)

$realIndexBefore = @(Invoke-GitCapture @(
    'diff', '--cached', '--name-only'
)).lines
if ($DryRun -and $realIndexBefore.Count -ne 0) {
    throw "The real Git index must be empty before a dry-run"
}

$savedIndex = $env:GIT_INDEX_FILE
$savedObjects = $env:GIT_OBJECT_DIRECTORY
$savedAlternates = $env:GIT_ALTERNATE_OBJECT_DIRECTORIES
try {
    if ($DryRun) {
        if ([string]::IsNullOrWhiteSpace($ScratchRoot)) {
            throw "ScratchRoot is required for a dry-run"
        }
        $scratch = [IO.Path]::GetFullPath($ScratchRoot)
        if (Test-Path -LiteralPath $scratch) {
            throw "Dry-run scratch path already exists: $scratch"
        }
        $null = New-Item -ItemType Directory -Path $scratch
        $objectDirectory = Join-Path $scratch 'objects'
        $null = New-Item -ItemType Directory -Path $objectDirectory

        $common = (Invoke-GitCapture @('rev-parse', '--git-common-dir')).lines[0]
        $common = Resolve-GitPath $common
        $env:GIT_INDEX_FILE = Join-Path $scratch 'index'
        $env:GIT_OBJECT_DIRECTORY = $objectDirectory
        $env:GIT_ALTERNATE_OBJECT_DIRECTORIES = Join-Path $common 'objects'

        $null = Invoke-GitCapture @('read-tree', 'HEAD')
        $null = Invoke-GitCapture (@('add', '--') + $expected)
    }

    $actual = @(
        (Invoke-GitCapture @('diff', '--cached', '--name-only')).lines |
            Where-Object { $_.Length -gt 0 } |
            ForEach-Object { $_.Replace('\', '/') } |
            Sort-Object -Unique
    )
    $compareArguments = @{
        ReferenceObject = $expected
        DifferenceObject = $actual
    }
    $difference = @(Compare-Object @compareArguments)
    if ($difference.Count -ne 0) {
        $details = @($difference | ForEach-Object {
            "$($_.SideIndicator) $($_.InputObject)"
        }) -join ', '
        throw "Staged paths do not match ${Layer}: $details"
    }

    if ($strict.Count -ne 0) {
        $whitespaceArguments = @('diff', '--cached', '--check', '--') + $strict
        $whitespace = Invoke-GitCapture -Arguments $whitespaceArguments
        if ($whitespace.lines.Count -ne 0) {
            throw "Staged whitespace check produced output: $($whitespace.lines -join '; ')"
        }
    }
}
finally {
    $env:GIT_INDEX_FILE = $savedIndex
    $env:GIT_OBJECT_DIRECTORY = $savedObjects
    $env:GIT_ALTERNATE_OBJECT_DIRECTORIES = $savedAlternates
}

$realIndexAfter = @(Invoke-GitCapture @(
    'diff', '--cached', '--name-only'
)).lines
if ($DryRun -and $realIndexAfter.Count -ne 0) {
    throw "Dry-run changed the real Git index"
}

$mode = if ($DryRun) { 'dry-run' } else { 'real-index' }
Write-Output (
    "PASS $mode ${Layer}: $($expected.Count) exact paths; " +
    "$($strict.Count) strict whitespace, $($exempt.Count) byte-preserved"
)
