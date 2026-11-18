# Compile an OpenVINO IR to an NPU37XX ELF blob with our own build of the NPU
# compiler (npu_ud_2025_44_rc1 = the driver's VCL 7.4), stage by
# stage, so any stage's MLIR can be inspected or edited and recompiled:
#
#   powershell -File vpuxc.ps1 IN.xml OUT.blob [-Opts "k=v k2=v2"] [-Profile] [-Keep]
#   powershell -File vpuxc.ps1 EDITED_vpuip.mlir OUT.blob      # resume from VPUIP
#
# Stages: IN.ie.mlir (vpux-translate --import-IE), IN.vpuip.mlir (vpux-opt
# --default-hw-mode=<opts>), IN.elf.mlir (--lower-VPUIP-to-ELF), OUT.blob
# (vpux-translate --export-ELF). Run the blob with `npubench blob OUT.blob`.
# -Profile adds DPU / DMA / SW task profiling (`npubench blob --profile`).
# -Keep keeps the intermediate .mlir files (they hold the weights: large).
param(
    [Parameter(Mandatory = $true, Position = 0)] [string] $In,
    [Parameter(Mandatory = $true, Position = 1)] [string] $Out,
    [string] $Opts = "",
    [switch] $Profile,
    [switch] $Keep
)
$ErrorActionPreference = 'Stop'
# NPUC_BIN: the compiler build's openvino\bin\intel64\Release (as for
# --npu-compiler); NPUC_TBB: its TBB, default the build tree's
# openvino\temp\Windows_AMD64\tbb\bin.
if (-not $env:NPUC_BIN) { throw 'set NPUC_BIN: the NPU compiler build (vpux-translate.exe, vpux-opt.exe)' }
$B = $env:NPUC_BIN
$T = if ($env:NPUC_TBB) { $env:NPUC_TBB } else { Join-Path $B '..\..\..\temp\Windows_AMD64\tbb\bin' }
$env:Path = "$T;$B;" + $env:Path
# NPUC_ARCH: the target flag (develop takes `--platform=NPU3720`).
$A = if ($env:NPUC_ARCH) { $env:NPUC_ARCH } else { '--vpu-arch=NPU37XX' }
$base = [IO.Path]::ChangeExtension($Out, $null).TrimEnd('.')
$o = "verify-locations=off"
if ($Profile) { $o += " profiling=true dpu-profiling=true dma-profiling=true sw-profiling=true" }
if ($Opts) { $o += " $Opts" }
# NPUC_OPTS: more default-hw options for every compile (e.g. from compile_lean.sh)
if ($env:NPUC_OPTS) { $o += " $env:NPUC_OPTS" }

function Step($what, [scriptblock] $cmd) {
    $t = Get-Date
    $log = & $cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        $log | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
        throw "$what failed (exit $LASTEXITCODE)"
    }
    Write-Host ("  {0,-12} {1,6:N1} s" -f $what, ((Get-Date) - $t).TotalSeconds)
}

$vpuip = "$base.vpuip.mlir"
if ($In.EndsWith('.xml')) {
    $ie = "$base.ie.mlir"
    $prof = @(); if ($Profile) { $prof = @('--vpux-profiling') }
    # task names (locations) must survive the printed MLIR between stages
    $dbg = @(); if ($Profile) { $dbg = @('--mlir-print-debuginfo') }
    Step 'import-IE' { & "$B\vpux-translate.exe" $A --import-IE @prof @dbg $In -o $ie }
    Step 'default-hw' { & "$B\vpux-opt.exe" $A "--default-hw-mode=$o" @dbg $ie -o $vpuip }
    if (-not $Keep) { Remove-Item $ie }
} else {
    $vpuip = $In
}
$elf = "$base.elf.mlir"
Step 'to-ELF' { & "$B\vpux-opt.exe" $A --lower-VPUIP-to-ELF $vpuip -o $elf }
Step 'export-ELF' { & "$B\vpux-translate.exe" $A --export-ELF $elf -o $Out }
if (-not $Keep) {
    Remove-Item $elf
    if ($In.EndsWith('.xml')) { Remove-Item $vpuip }
}
Write-Host ("  -> {0} ({1:N1} MB)" -f $Out, ((Get-Item $Out).Length / 1MB))
