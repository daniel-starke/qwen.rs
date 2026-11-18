# Run a command while holding the machine-wide NPU lock (the named mutex
# `Global\qwen-rs-npu`, the one `crates/ov/src/lock.rs` takes in the NPU
# engine). Use it for anything that compiles for or runs on the NPU outside
# `cli`, such as the Python experiments in this directory:
#
#   powershell -File crates/ov/scripts/npu_lock.ps1 python crates/ov/scripts/npu_compile.py 16 1024
#
# The kernel releases the mutex if this process dies, so a kill cannot leave
# it stuck.
$m = New-Object System.Threading.Mutex($false, 'Global\qwen-rs-npu')
try {
    $got = $false
    try { $got = $m.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
    if (-not $got) {
        Write-Host "npu_lock: waiting for another process to release the NPU"
        $t = Get-Date
        try { [void]$m.WaitOne() } catch [System.Threading.AbandonedMutexException] {}
        Write-Host ("npu_lock: got the NPU after {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)
    }
    $cmd = $args[0]
    $rest = @()
    if ($args.Count -gt 1) { $rest = $args[1..($args.Count - 1)] }
    & $cmd @rest
    $code = $LASTEXITCODE
} finally {
    try { $m.ReleaseMutex() } catch {}
    $m.Dispose()
}
exit $code
