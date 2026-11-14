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

Open lab 1 → Practice → Install local packages. The site creates a `.venv` virtual environment inside the project and installs the pinned course packages. This downloads several hundred megabytes or more. Watch the real terminal output until it says `succeeded`.

Run the environment diagnostic. Then move to lab 2. Model weights download separately on first use. The first workbench example uses only built-in Python and can run before package installation.

No PowerShell activation script is needed. No execution-policy change is needed.

### Manual package installation if preferred

From the folder containing `server.py`, in PowerShell, after setting `$python312` to a working Python 3.12 executable:

```powershell
& $python312 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r .\lab\requirements.txt -r .\lab\requirements-foundations.txt
& .\.venv\Scripts\python.exe .\lab\lab.py doctor
```

The matching Command Prompt commands are:

```bat
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r lab\requirements.txt -r lab\requirements-foundations.txt
.venv\Scripts\python.exe lab\lab.py doctor
```

## Upgrade an existing working installation

Close the running lab with Control+C. Extract the new archive into a separate folder first. Keep a backup of your existing project. Copy the new source files and the `public`, `course`, `tests`, and `lab` folders into your existing website folder, merging folders rather than deleting the old ones. The release archive does not contain your `.venv`, `lab/outputs`, `lab/inputs`, or `runs`; keep those directories. If you previously edited course source files, preserve your edits before replacing those files.

Your lesson progress stays in the same browser profile at http://localhost:8765. Use that same address after upgrading. Export evidence notes and terminal logs when you need a portable copy. Browser progress does not move automatically between ports, browser profiles, or computers.

If the website already uses its own working `.venv`, add the new packages from PowerShell in the folder containing `server.py`:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r .\lab\requirements-foundations.txt
& .\.venv\Scripts\python.exe .\server.py
```

If you still use the original `sml-lab-zero-to-hero` environment shown in your diagnostic, reuse its direct executable. Set this in the **same PowerShell window** that launches the website, adjusting the path only if your folder differs:

```powershell
$env:SML_LAB_PYTHON = "$env:USERPROFILE\Downloads\sml-lab-zero-to-hero\sml-lab-zero-to-hero\.venv\Scripts\python.exe"
Test-Path $env:SML_LAB_PYTHON
& $env:SML_LAB_PYTHON -m pip install -r .\lab\requirements-foundations.txt
& $env:SML_LAB_PYTHON .\server.py
```

Continue only if `Test-Path` prints `True`. The environment variable tells the website which executable to use for every lab. It avoids the missing `py` launcher and the Microsoft Store `python` alias. It lasts for this terminal session. This add-on installation preserves the installed PyTorch build.

The matching Command Prompt commands, from the website folder:

```bat
set "SML_LAB_PYTHON=%USERPROFILE%\Downloads\sml-lab-zero-to-hero\sml-lab-zero-to-hero\.venv\Scripts\python.exe"
"%SML_LAB_PYTHON%" -m pip install -r lab\requirements-foundations.txt
"%SML_LAB_PYTHON%" server.py
```

In the website, filter the learning path to **Foundations**. Start with **Understand how models learn**, then **Choose a model and design the evidence**, then **Linear regression**. Run both experiments, load a saved example, change an input, predict, and export your checkpoint notes. The new builds use bundled or generated data, so after installing packages they do not need model downloads or a cloud account.

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
