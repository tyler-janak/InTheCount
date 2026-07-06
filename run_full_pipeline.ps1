# run_full_pipeline.ps1
# ---------------------------------------------------------------------------
# One-button full pipeline.
#
# Flags:
#   -SkipTrain      : skip the slow tuning retrain (uses on-disk models)
#   -PromoteGame    : promote the game-ML candidate to production (with backup)
#   -Push           : git push at the end
#   -Adapter "Name" : Wi-Fi adapter name for the DNS lock (default "Wi-Fi")
#
# Usage:
#   .\run_full_pipeline.ps1
#   .\run_full_pipeline.ps1 -SkipTrain
#   .\run_full_pipeline.ps1 -PromoteGame -Push

[CmdletBinding()]
param(
    [switch]$SkipTrain,
    [switch]$PromoteGame,
    [switch]$Push,
    [string]$Adapter = "Wi-Fi"
)

$ErrorActionPreference = "Continue"
$here = "C:\Users\tyler\OneDrive\Documents\Machine Learning\MLB Betting Website"
Set-Location $here
$env:BULLPEN_RETRAIN = "skip"

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
$sample = "outputs\hitterspitchers_2026-05-15.csv"
if (Test-Path $sample) {
    $sz = (Get-Item $sample).Length
    Write-Host ("  {0,-45} {1,8} bytes" -f $sample, $sz)
}

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
nslookup statsapi.mlb.com 1.1.1.1 | Select-String "Address" | Select-Object -First 2

# --- 3. Refresh data --------------------------------------------------------
Section "Refresh per-game feature tables"
python refresh_full_history.py --skip-2026

# --- 4. Retrain player models -----------------------------------------------
if (-not $SkipTrain) {
    Section "Retrain pitcher + hitter models (tuning + calibration ON)"
    Write-Host "  (slow: ~20-40 min; pass -SkipTrain to skip)"
    python hitterspitchers_train.py
}
else {
    Section "Skipping model retrain (-SkipTrain flag set)"
}

# --- 5. Game-ML CANDIDATE (never auto-promotes by default) -----------------
Section "Train game-ML candidate (won't touch betting_model.pkl)"
python train_game_model.py
if ($PromoteGame) {
    Section "Promote game-ML candidate to production"
    python train_game_model.py --promote
}

# --- 6. Force-backfill historical snapshots ---------------------------------
Section "Force-backfill historical snapshots with new models + weights"
Write-Host "  (slow: ~20-30 min; hits MLB Stats API for past lineups)"
python -c "from backfill_player_predictions import backfill; backfill(start='2026-03-25', end=None, force=True, grade=False, verbose=True)"

# --- 7. Fill blend components + recompute proj_* per current weights -------
Section "Fill blend components + recompute proj_* (fast: ~1-2 min)"
python backfill_blend_components.py --force

# --- 8. Grade ---------------------------------------------------------------
Section "Grade player projections (no MLB API surface)"
python -c "from grade_player_predictions import grade_player_predictions; grade_player_predictions(snapshots_dir='outputs', output_file='2026_player_accuracy.csv', season_start='2026-03-25')"

Section "Daily run (game ML + NRFI + props grading; API retries built in)"
python daily_update.py

# --- 9. Empirical blend tuning ---------------------------------------------
Section "Empirical alpha grid search per stat"
python tune_blend.py
python tune_blend.py --since 2026-05-15

# --- 10. Dashboard ---------------------------------------------------------
Section "Unified dashboard"
python print_all_stats.py

if ($Push) {
    Section "Push to GitHub"
    git pull origin main --no-rebase -X ours --no-edit
    git add -A
    git commit -m "Full pipeline rerun + retrain (incl. Total Bases)"
    git push
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
if (-not $PromoteGame) {
    Write-Host ""
    Write-Host "  If train_game_model showed positive 'Delta vs prod', re-run with -PromoteGame to ship it." -ForegroundColor Yellow
    Write-Host "  Otherwise production model stays untouched." -ForegroundColor Yellow
}
if (-not $Push) {
    Write-Host "  Add -Push to commit + push at the end." -ForegroundColor Yellow
}
