# run_full_pipeline.ps1
# ---------------------------------------------------------------------------
# One-button full pipeline (local, Windows).
#
# Flags:
#   -SkipTrain      : skip the player-model retrain (uses on-disk models)
#   -SkipEval       : skip the out-of-sample backtest + evaluation + report
#   -PromoteGame    : refit the game model on all completed games and promote it
#   -Push           : git push at the end
#   -Adapter "Name" : Wi-Fi adapter name for the DNS lock (default "Wi-Fi")
#
# Usage:
#   .\run_full_pipeline.ps1
#   .\run_full_pipeline.ps1 -SkipTrain -SkipEval
#   .\run_full_pipeline.ps1 -PromoteGame -Push

[CmdletBinding()]
param(
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$PromoteGame,
    [switch]$Push,
    [string]$Adapter = "Wi-Fi"
)

$ErrorActionPreference = "Continue"
$here = "C:\Users\tyler\OneDrive\Documents\Machine Learning\MLB Betting Website"
Set-Location $here
$env:BULLPEN_RETRAIN = "skip"
$py = ".\.venv\Scripts\python.exe"

function Section($title) {
    Write-Host ""
    Write-Host "==== $title " -ForegroundColor Cyan -NoNewline
    $pad = 70 - $title.Length
    if ($pad -lt 0) { $pad = 0 }
    Write-Host ("=" * $pad) -ForegroundColor Cyan
}

# --- 1. Restore snapshots in case prior runs zero-byte'd them ---------------
Section "Restore snapshots from git (if a prior run blew them out)"
git checkout HEAD -- outputs/ 2>$null

# --- 2. Lock DNS so MLB Stats API cannot go unreachable mid-run -------------
Section "Lock DNS to Cloudflare + Google and flush cache"
try {
    Set-DnsClientServerAddress -InterfaceAlias $Adapter -ServerAddresses ("1.1.1.1","8.8.8.8") -ErrorAction Stop
    Write-Host "  DNS pinned on $Adapter -> 1.1.1.1, 8.8.8.8"
}
catch {
    Write-Host "  (could not set DNS on $Adapter - check Get-NetAdapter)" -ForegroundColor Yellow
}
ipconfig /flushdns | Out-Null

# --- 3. Statcast (previous two seasons + current) + validation + features ---
Section "Collect / validate Statcast and rebuild windowed feature tables"
& $py refresh_full_history.py

# --- 4. Retrain player models (production: validation-selected, refit on all)
if (-not $SkipTrain) {
    Section "Retrain pitcher + hitter models"
    & $py hitterspitchers_train.py
}
else {
    Section "Skipping model retrain (-SkipTrain flag set)"
}

# --- 5. Out-of-sample evaluation (never touches production models) ---------
if (-not $SkipEval) {
    Section "Backtest: game model (train/validation/test)"
    & $py train_game_model.py
    Section "Backtest: player models (train/validation/test)"
    & $py run_backtest.py
    Section "Evaluate saved predictions, figures, metrics, report"
    & $py evaluate_models.py
}

# --- 6. Game model promotion ------------------------------------------------
if ($PromoteGame) {
    Section "Refit game model on all completed games and promote"
    & $py train_game_model.py --production --promote
}

# --- 7. Daily run (game picks + player snapshots + grading + rankings) ------
Section "Daily run"
& $py daily_update.py

# --- 8. Dashboard ---------------------------------------------------------
Section "Unified dashboard"
& $py print_all_stats.py

if ($Push) {
    Section "Push to GitHub"
    git pull origin main --no-rebase -X ours --no-edit
    git add -A
    git commit -m "Full pipeline rerun (history window + evaluation)"
    git push
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
