# Windows — PowerShell or Command Prompt

[Wiki home](Home.md) · [Repository README](../../README.md)

Start with [Getting started](Getting-Started.md) to download the project and check requirements. Prefer a [dedicated VM](Running-in-a-VM.md) when its guest platform supports the pinned packages.

### 1. Install Python

Run this from **PowerShell or Command Prompt**:

```powershell
winget install --exact --id Python.Python.3.12 --source winget --scope user
```

Close the terminal application and open it again after installation. If `winget` is unavailable, follow [Microsoft's Python setup guide](https://learn.microsoft.com/en-us/windows/python/beginners) or use the [official Python downloads](https://www.python.org/downloads/windows/). Python 3.12.10 was the final 3.12 release with the traditional full maintenance installers; later 3.12 security releases exist. Do not interpret that installer version as the latest security patch.

### 2. Launch the website

In File Explorer, double-click **`start-lab.cmd`** and keep its window open. It finds common Python installations and an existing project environment, so a missing `py` launcher does not prevent startup.

Or, from **PowerShell** in the extracted project folder:

```powershell
.\start-lab.cmd
```

Or, from **Command Prompt** in that folder:

```bat
start-lab.cmd
```

Open **http://localhost:8765** if the browser does not open automatically. Continue with [Run your first experiment](First-Experiment.md).

### Missing `py`, Microsoft Store alias, or custom installation

`py` is a separate launcher. The `python` command may point to a Microsoft Store shortcut rather than your installed interpreter. Use the actual executable directly.

**PowerShell** — the ampersand executes the path stored in the variable:

```powershell
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
Test-Path $python312
& $python312 --version
& $python312 .\server.py
```

Continue only if `Test-Path` prints `True` and the version is 3.12.x. A system-wide installation may be under `C:\Program Files\Python312\python.exe`; use the actual path. Setting `$python312` does not create a command named `python312`.

**Command Prompt:**

```bat
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" server.py
```

### Optional manual dependency installation

The website can install these packages for you. To do it manually, in **PowerShell** after setting `$python312` above:

```powershell
& $python312 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r .\lab\requirements.txt -r .\lab\requirements-foundations.txt
& .\.venv\Scripts\python.exe .\lab\lab.py doctor
& .\.venv\Scripts\python.exe .\server.py
```

The equivalent **Command Prompt** commands:

```bat
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r lab\requirements.txt -r lab\requirements-foundations.txt
.venv\Scripts\python.exe lab\lab.py doctor
.venv\Scripts\python.exe server.py
```

No PowerShell activation script or execution-policy change is needed. See [WINDOWS-SETUP.md](../../WINDOWS-SETUP.md) for upgrades and custom environments.
