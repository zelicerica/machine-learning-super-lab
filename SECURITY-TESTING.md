# Security test candidate — read first

GitHub release: **2026.11-transformer.2**, combining the newer Lesson 59 Transformer explorer with the owner-tested setup-fix ZIP **2026.11-midnight.4-security-test.3**. Historical ZIP checks below retain their original scope. This is a separate **Windows x64, Python 3.12, CPU-only functionality-test package**, not certification that every model works or that all malicious code is absent. Do not extract over your running lab or point this copy at its Python environment.

This revision fixes browser setup in normal long Downloads paths. Setup creates a fresh `.lab-environments/env-...` environment using Windows extended-length paths, checks the packages, then selects it through `.runtime-python`. Existing environments and earlier runtime configuration are preserved. No administrator rights, Windows policy change or security-warning bypass is required. Do not dismiss an unexplained Windows/browser security warning; record its exact message for diagnosis.

The glass-cube video now defaults to its original brightness on every page. **Dim video** in the top bar restores the previous dimmed reading treatment; the button then says **Original video** to switch back. The choice survives reloads and navigation and is independent of Play/Pause. Both modes use the same unmodified MP4. The simulated DOM check passed; a new visual browser check was unavailable. This revision changes only frontend controls/styles and the release label; the earlier security/package/model results below remain their recorded checks.

The localhost protections passed targeted and regression tests. The dependency upgrades passed imports, core regression tests and a tiny random-model adapter round trip. Actual pretrained SmolLM2 inference/training, SmolVLM vision, GPU, macOS/Linux and a visual browser review have **not** been rerun with these versions. Test these before choosing to switch your working lab. No paid compute is needed.

## What changed

- All server modes bind to loopback. Preview cannot execute Python. Host checks retain the actual port and apply in preview; exact HTTP Origin and cross-site Fetch Metadata checks supplement the existing random session token, JSON request and body limits.
- Full setup creates a fresh environment without modifying an existing `.venv` or managed environment, supports long Windows paths and backs up earlier runtime configuration. Cancellation checks Windows taskkill's return code and falls back to terminating the worker.
- The optional tqdm Telegram helper is removed after package installation, as requested by the owner; normal tqdm is retained. Its absence deliberately differs from the original package RECORD. A later reinstall can restore it; rerun the removal command afterward.
- Candidate versions: PyTorch 2.13.0+cpu, Transformers 5.23.0, Parameter-Efficient Fine-Tuning (PEFT) 0.21.3, Accelerate 1.15.0, Pillow 13.0.0, pip 26.3, setuptools 84.2.1, Mermaid 12.3.0. NumPy 2.2.6, scikit-learn 1.7.2, SciPy 1.16.3 and Matplotlib 3.10.7 retained. All resolved Windows dependencies and their exact wheel hashes are in `lab/requirements-windows-cpu.lock`.
- The vision entry point uses Transformers' current `AutoModelForImageTextToText` class. That class imports; real image-model execution remains unverified. Windows UTF-8 reads/launcher setting and two content-test reads were corrected.

## Test without touching the live installation

1. Extract the complete ZIP into a **new folder**, including an ordinary Downloads folder if preferred. The ZIP contains source and evidence, not a portable virtual environment, downloaded model weights, or learner outputs.
2. Run **`start-security-test.cmd`**, open **http://localhost:8766**, and choose **Install local packages** in the workbench or Practice tab. Confirm the lab's normal installation dialog. Setup creates and validates a new environment; follow its actual log in the console. If installation fails, earlier environments remain available; a retry creates another fresh environment. Failed partial environments are retained for diagnosis, not automatically deleted.

The equivalent manual setup from PowerShell is:

```powershell
py -3.12 lab/setup_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Installation failed; keep the live lab unchanged.' }
$labPython = (Get-Content .runtime-python -Raw).Trim()
& $labPython -m pip check
```

For Command Prompt:

```bat
py -3.12 lab\setup_environment.py
if errorlevel 1 exit /b 1
for /f "usebackq delims=" %P in (".runtime-python") do @"%P" -m pip check
```

Installation fetches public registry wheels; no package source builds are allowed. PyTorch's CPU wheel uses its official download URL; other artifacts are pinned to registry hashes. The exact fresh online installation from the extracted ZIP was **not** run during this review. Tested installation used the already downloaded verified wheels and an isolated environment. Python 3.12.10 was the available test interpreter; its operating-system/OpenSSL patch level is a separate residual risk, not updated by this ZIP.

3. Confirm the security-test release, new folder and selected interpreter at `/api/workspace`. The test launcher ignores an inherited `SML_LAB_PYTHON` override and lets this copy select its own validated runtime. Keep your existing lab on 8765 running if desired. Port 8766 has separate browser storage; existing notes/progress at 8765 remain there. Do not clear site data.

4. In a second PowerShell window, run the offline checks. Use a new output name on every smoke-test run:

```powershell
$labPython = (Get-Content .runtime-python -Raw).Trim()
& $labPython -X utf8 -m unittest discover -s tests -v
& $labPython -X utf8 -m unittest discover -s lab/tests -v
& $labPython -X utf8 tests/security_compatibility.py --output lab/outputs/security-compat-01
```

The last command checks tiny random-model training, independent synthetic-token evaluation, safetensors saving, frozen-base preservation, adapter reload and inference on a new token input. It saves a report, input contract and artifact hashes; it is **not pretrained inference** or a useful language-quality benchmark.

5. Check the interface: navigate lessons/references, render diagrams, edit/run a small workbench example, cancel a running exercise, train a Foundations linear model into a fresh output directory, load its saved example and predict a changed input, then inspect Results. Never paste untrusted code into the workbench.

6. To test the actual language/vision paths, use fresh test-only model cache and output directories. Downloads are required if weights are absent. These commands are **manual checks not yet executed for this candidate**:

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'test-model-cache'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$labPython = (Get-Content .runtime-python -Raw).Trim()
Push-Location lab
& $labPython lab.py doctor
& $labPython lab.py infer --prompt 'Explain a neural network briefly.' --max-new-tokens 24
& $labPython lab.py eval --split validation --output outputs/security-base-validation.json
& $labPython lab.py train --steps 60 --output outputs/security-adapter
& $labPython lab.py infer --adapter outputs/security-adapter --prompt 'Explain a neural network briefly.' --max-new-tokens 24
& $labPython lab.py eval --adapter outputs/security-adapter --split validation --output outputs/security-adapter-validation.json
# Final independent test: run after choosing settings; do not tune against it.
& $labPython lab.py eval --adapter outputs/security-adapter --split test --output outputs/security-adapter-test.json
# Replace the image path with a non-sensitive test PNG or JPEG.
& $labPython vision_lab.py 'C:\path\to\test-image.png' --prompt 'Describe the image briefly.'
Pop-Location
```

Save console output and reports. A package/API failure is a reason to keep the original installation and report the actual traceback, not to silently downgrade dependencies. Do not rent a GPU for these small CPU checks.

## Backup before any future switch

Nothing in this package automatically promotes or overwrites your current lab. First stop both servers with Ctrl+C and wait for jobs to finish. Keep the entire original directory and its environment at their original paths.

In PowerShell, set `$live` to your original folder. This makes a new private source/data backup; it deliberately leaves the original virtual environment in place for rollback:

```powershell
$live = (Resolve-Path (Read-Host 'Full path of your original live lab folder')).Path
$backup = Join-Path (Split-Path $live -Parent) ('SML-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
if (Test-Path -LiteralPath $backup) { throw 'Choose a new backup path.' }
robocopy $live $backup /E /COPY:DAT /DCOPY:DAT /XJ /XD .venv venv .lab-environments security-review node_modules __pycache__ .cache
if ($LASTEXITCODE -ge 8) { throw 'Backup failed; do not switch.' }
```

Verify that `lab/inputs`, `lab/outputs`, `runs`, personal notes and any nonstandard learner folders are present in the backup. Preserve any notes stored elsewhere separately. Do not publish this backup; it may contain private data or configuration.

Browser progress/notes live in your browser profile at the exact origin. Keep that profile and `http://localhost:8765`. To save the existing progress JSON privately, open Developer Tools → Console **on your original localhost:8765 page** and execute `copy(localStorage.getItem('super-lab-v1'))`, then paste the clipboard into a private `browser-progress.json` backup. Do not clear storage. Theme/motion preferences are separate `super-lab-*` keys and may also be copied if wanted. No browser-storage export was performed by the reviewer.

## Optional promotion, restart and rollback — instructions only

After all desired functionality passes, extract this ZIP again into a fresh folder (for example `C:\SML-sec-live`), create its **own** environment with browser setup or the setup helper above. Do not reuse the old environment. Copy learner directories only into absent destinations; never merge over existing results:

```powershell
$new = 'C:\SML-sec-live'  # Folder containing this package's server.py
foreach ($relative in @('lab\inputs','lab\outputs','runs')) {
    $source = Join-Path $live $relative
    $destination = Join-Path $new $relative
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination. Resolve conflicts manually." }
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $destination -Recurse }
}
```

Copy/verify personal notes outside those directories individually. Existing job logs can refer to old paths. Treat saved joblib/TorchScript models as executable trusted artifacts; cross-version reload compatibility of your own saved models must be tested on copies. Keep originals unchanged.

Stop the original launcher, then start the new folder at the original origin:

```powershell
Set-Location $new
$env:SML_LAB_PYTHON = (Get-Content (Join-Path $new '.runtime-python') -Raw).Trim()
.\start-lab.cmd --host 127.0.0.1 --port 8765 --no-browser
```

Check `/api/workspace`, refresh the page and verify progress/notes. If anything fails, Ctrl+C in the new launcher, return to `$live`, set `SML_LAB_PYTHON` to `$live\.venv\Scripts\python.exe`, and run the same `start-lab.cmd --host 127.0.0.1 --port 8765 --no-browser`. Your original folder/environment and browser storage remain available. Retain any new outputs from the candidate separately; do not overwrite original files during rollback.

## Residual risks and monitoring

The workbench executes code with your user permissions; it is not an operating-system sandbox. Use only trusted code and model artifacts. Loopback is not protection against other malicious local processes. Package-registry authenticity and advisory scans do not prove a package cannot be malicious. Original installed PyTorch had RECORD integrity checking but was not compared against its original downloaded wheel. The original prebuilt JavaScript bundle has incomplete exact transitive provenance; `security/javascript-inventory.json` marks that gap.

Monitor actual job failures, model reload mismatches, changed predictions, dependency advisories and unexpected network activity using your existing OS tools. Stop the test server on unexpected behavior and retain its logs. Back up each new experiment and use fresh output names. Recheck current advisories before a later promotion; saved scan results describe the review date only. No live replacement, GitHub push, public exposure or paid resource was performed.
