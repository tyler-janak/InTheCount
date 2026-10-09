# sync_to_github.ps1 - push this local project to GitHub (tyler-janak/inthecount)
#
#   Local code wins.  GitHub's newer DAILY DATA wins (accuracy logs, live
#   predictions, dated snapshots, rankings JSON) because the Actions cron kept
#   writing those after this copy was last pulled.  GitHub's index.html edits
#   (FAQ / Power Rankings hidden, betting FAQ removed) are kept too.
#
#   player_rankings.py exists in two different versions:
#     GitHub (Aug 26 "Update Player Rankings", what the site runs now)  <- default
#     local  (Aug 21-25 WAR / positional work that was never pushed)
#   Run with  -KeepLocalRankings  to push the local one instead.
#
# Usage (PowerShell, in the project folder):
#   powershell -ExecutionPolicy Bypass -File .\sync_to_github.ps1
#   powershell -ExecutionPolicy Bypass -File .\sync_to_github.ps1 -KeepLocalRankings
#   add -DryRun to stop before the push and inspect with `git log` / `git diff HEAD~1`
param([switch]$KeepLocalRankings, [switch]$DryRun)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Run($cmd) { Write-Host "> $cmd" -ForegroundColor DarkGray; Invoke-Expression $cmd; if ($LASTEXITCODE) { throw "failed: $cmd" } }

if ((git rev-parse --abbrev-ref HEAD) -ne "main") { throw "Switch to the main branch first." }
$stamp = Get-Date -Format "yyyyMMdd-HHmm"

# 1. Workflow: install the updated daily job (more run times, next-day slate,
#    3-season caches) and drop the two NRFI workflows whose script no longer exists.
Copy-Item docs\daily.yml.proposed .github\workflows\daily.yml -Force
foreach ($wf in ".github/workflows/nrfi_rebackfill.yml", ".github/workflows/retrain_nrfi.yml") {
  if (Test-Path $wf) { git rm -q -- $wf }
}

# 2. Commit everything local (moved/retired files are recorded as deletions).
Run "git add -A"
git diff --cached --quiet
if ($LASTEXITCODE) {
  Run "git commit -q -m 'Audit: 2024-26 history window, leakage fixes, evaluation + report, stacking, lineup features, next-day slate'"
}
Run "git branch backup/local-$stamp"
Write-Host "Safety copy of your local state: branch backup/local-$stamp" -ForegroundColor Green

# 3. Bring in GitHub's history and resolve overlaps by rule.
Run "git fetch origin"
git merge origin/main --no-commit --no-ff -m "Merge GitHub daily data + site edits into local" 2>&1 | Out-Host
$theirs = '^(2026_.*\.csv|mlb_pick_log.*|calibration\.json|index\.html|data/player_ages\.json)$'
if (-not $KeepLocalRankings) { $theirs = $theirs.TrimEnd('$').TrimEnd(')') + '|player_rankings\.py)$' }
$conflicts = @(git diff --name-only --diff-filter=U)
foreach ($f in $conflicts) {
  $useTheirs = ($f -match $theirs) -or ($f -match '^outputs/' -and $f -notmatch '^outputs/model_evaluation/')
  $side = if ($useTheirs) { "--theirs" } else { "--ours" }
  git checkout $side -- "$f" 2>$null
  if ($LASTEXITCODE) { git rm -q -- "$f" } else { git add -- "$f" }
  Write-Host ("  {0,-8} {1}" -f ($(if ($useTheirs) {"GitHub"} else {"local"}), $f))
}
# files GitHub changed that did NOT conflict but should still be local:
if ($KeepLocalRankings) { git checkout HEAD -- player_rankings.py; git add player_rankings.py }
$left = @(git diff --name-only --diff-filter=U)
if ($left.Count) { throw "Unresolved files: $left  (nothing pushed; 'git merge --abort' undoes the merge)" }
Run "git commit -q --no-edit"

# 4. Quick checks before anything leaves your machine.
& .venv\Scripts\python.exe -m pytest -q tests\test_slate_switch.py tests\test_stacking.py
if ($LASTEXITCODE) { throw "Checks failed - nothing pushed. 'git reset --hard backup/local-$stamp' restores your local state." }

if ($DryRun) { Write-Host "Dry run: merged locally, NOT pushed. Push later with: git push origin main" -ForegroundColor Yellow; exit 0 }
Run "git push origin main"
Write-Host "Pushed. Render redeploys the site from this commit; the next daily run uses the new workflow." -ForegroundColor Green
