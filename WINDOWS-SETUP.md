> **CURRENT WINDOWS CPU UPDATE — Python 3.12.** Read [SECURITY-TESTING.md](SECURITY-TESTING.md) first. Use a new folder and its own environment on localhost port 8766. Earlier platform and dependency recipes below are historical; use the hashed Windows lock for this candidate. Pretrained language/vision and GPU compatibility are not yet certified.

# Run Machine Learning Super Lab on Windows

Use PowerShell or Command Prompt. Git Bash, Git, Node.js, and a cloud account are not required. The browser site runs actual Python programs on your computer; it is not an isolated virtual machine.

## 1. Install Python 3.12

Open PowerShell from Start and run:

```powershell
winget install --exact --id Python.Python.3.12 --source winget --scope user
```

Let installation finish. Close the entire terminal application and open PowerShell again. The course uses Python 3.12 for compatibility with its pinned package versions.

## 2. Find Python without depending on the launcher

The `py` command is a separate launcher. If it is missing, Python may still be installed. These PowerShell commands find common installation locations:

```powershell
$python312 = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    "$env:ProgramFiles\Python312\python.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($python312) {
    & $python312 --version
} else {
    winget list --exact --id Python.Python.3.12
}
```

Expected: `Python 3.12.x`. The ampersand tells PowerShell to execute the program at the quoted path. The variable lasts only in this PowerShell session.

If Python appears in the package list but neither path exists, check Settings → Apps → Installed apps for Python 3.12, and use its actual installation path. To inspect executable commands available on your search path:

```powershell
Get-Command python, python3, py -All -ErrorAction SilentlyContinue
```

If no installed package is found, run step 1 and check its full output. If `winget` is missing, install or update Microsoft's App Installer from the Microsoft Store, reopen PowerShell, and retry. Alternatively use the official Python installer linked below. For the classic Python installer, select 64-bit Python 3.12, include the launcher, and select “Add python.exe to PATH.”

## 3. Extract and launch the website

Download `SML-Lab-Localhost.zip`. In File Explorer, right-click it → Extract All. Open the extracted `SML-Lab-Localhost` folder and double-click `start-lab.cmd`.

The launcher checks the Python launcher, common direct executable paths, then the `python` command. Keep its window open. It opens:

http://localhost:8765

If the browser does not open, enter that address yourself. This address works after the server starts on your own computer.

### Launch from PowerShell instead

Change the example folder to the location containing `server.py`:

```powershell
Set-Location "$env:USERPROFILE\Downloads\SML-Lab-Localhost\SML-Lab-Localhost"
.\start-lab.cmd
```

If you extracted directly to Downloads, the folder might instead be `Downloads\SML-Lab-Localhost`. Verify with `Get-ChildItem`; `server.py` and `start-lab.cmd` must appear. With a custom Python location:

```powershell
$python312 = "C:\your\actual\Python312\python.exe"
& $python312 .\server.py
```

### Launch from Command Prompt instead

```bat
cd /d "%USERPROFILE%\Downloads\SML-Lab-Localhost\SML-Lab-Localhost"
start-lab.cmd
```

The same extraction-path rule applies. To use a common direct Python path in Command Prompt:

```bat
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" server.py
```

## 4. Install the machine learning packages from the website

**Browser users: complete Lesson 21 — Build your workspace — first.** Open **Practice → Install local packages** and wait for **SUCCEEDED**. Setup creates a fresh `.lab-environments/env-...` environment, installs the hash-pinned Windows CPU packages, validates them and selects the runtime through `.runtime-python`. Normal Downloads paths are supported, and existing environments are preserved. This downloads several hundred megabytes or more; watch the actual console log.

Run the environment diagnostic. Then return to Foundations Lesson 1 or your chosen learning path. Model weights download separately on first use. The first workbench example uses only built-in Python and can run before package installation.

No PowerShell activation script is needed. No execution-policy change is needed.

### Manual package installation if preferred

From the folder containing `server.py`, use the same setup helper as the browser:

```powershell
py -3.12 lab/setup_environment.py
$labPython = (Get-Content .runtime-python -Raw).Trim()
& $labPython -m pip check
```

In Command Prompt:

```bat
py -3.12 lab\setup_environment.py
if errorlevel 1 exit /b 1
start-lab.cmd
```

If `py` is unavailable, replace it with the verified Python 3.12 executable described above. Setup never needs an administrator or an execution-policy change.

## Upgrade an existing working installation

Extract into a new folder and keep the old folder, environment, learner inputs/outputs, run logs and notes. For the updated dependencies, use Lesson 21 or the setup helper to create a fresh environment rather than modifying a working one in place. For side-by-side testing use `start-security-test.cmd` and http://localhost:8766. Read [SECURITY-TESTING.md](SECURITY-TESTING.md) for exact backup, promotion, restart and rollback commands.

Browser progress belongs to the browser profile and exact origin. Keep http://localhost:8765 when you eventually switch copies; do not clear its storage. Port 8766 has separate progress. Copy learner files only into empty destinations and keep the originals.

## Troubleshooting

For the synthetic-data addition, open **Build and retrieve → Synthetic data: generate, audit, and compare** (lesson 42). The same environment and foundation packages work; no new package family is required. Run the four Practice comparisons, load the saved example, and inspect the plots in Experiment results. The synthetic-data reference is also in the sidebar.

- **`py` is not recognized:** use the direct executable or the included launcher. Installing the separate launcher is optional.
- **`python` opens Microsoft Store:** that command may resolve to Windows' shortcut. Use the direct executable. Disabling a shortcut does not install Python.
- **Port 8765 is already in use:** first check whether the lab is already running. To select another port: `& $python312 .\server.py --port 8766`, then open http://localhost:8766.
- **Missing package:** install local packages inside the website, then run the diagnostic again. The website uses the project environment automatically.
- **Package download fails:** keep the full terminal log. Check internet access and available disk space. Retry package setup; package installation can reuse downloads.
- **Close the lab:** save logs you need, then press Control+C in the launcher window. Stopping a job can leave partial output; start a new training run with a new output folder.
- **Custom Python path:** provide it explicitly as shown above; do not paste a guessed path.

Official references:

- https://learn.microsoft.com/en-us/windows/python/beginners
- https://docs.python.org/3.12/using/windows.html
- https://www.python.org/downloads/release/python-31210/ (classic installer; later Python 3.12 security releases also exist)
- https://github.com/microsoft/winget-pkgs/tree/master/manifests/p/Python/Python/3/12
