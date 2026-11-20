# Push a snapshot of the committed code to a Hugging Face Docker Space.
# Usage: .\deploy\push_to_hf.ps1 -Space <hf-username>/<space-name>
# Builds a fresh single-commit repo in a temp folder, so your checkout and its history are untouched.
# Only committed files are pushed; commit first if you want local changes included.
param([Parameter(Mandatory = $true)][string]$Space)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("hf-space-" + [guid]::NewGuid().ToString("N"))

git clone --quiet --no-local $root $tmp
if (-not $?) { throw "git clone failed" }
Push-Location $tmp
try {
    git lfs pull
    Remove-Item -Recurse -Force .git
    # The Space needs its own README with the Docker SDK front matter.
    Copy-Item deploy\hf_space_README.md README.md -Force

    git init --quiet -b main
    git lfs install --local
    git add -A
    git commit --quiet -m "Deploy StructuralAI backend"
    git push --force "https://huggingface.co/spaces/$Space" main
    if (-not $?) { throw "push failed (log in with 'huggingface-cli login' or use an access token as the password)" }
    Write-Host "Pushed. Build logs: https://huggingface.co/spaces/$Space"
}
finally {
    Pop-Location
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
