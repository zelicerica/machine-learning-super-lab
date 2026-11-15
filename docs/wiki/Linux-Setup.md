# Linux — Ubuntu 24.04 example

[Wiki home](Home.md) · [Repository README](../../README.md)

Start with [Getting started](Getting-Started.md) to download the project and check requirements. Prefer a [dedicated VM](Running-in-a-VM.md) when its guest platform supports the pinned packages.

The tested model runtime uses 64-bit Linux with Python 3.12 and CPU PyTorch. The following package commands are specifically for **Ubuntu 24.04**, which supplies Python 3.12:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
python3.12 --version
```

On another distribution, install Python 3.12 and its virtual-environment support with that distribution's supported packages or a trusted Python version manager. Do not assume every `apt`, `dnf`, or `pacman` repository contains the same version. Keep the system Python intact.

Change into the extracted project folder and launch:

```bash
cd ~/Downloads/machine-learning-super-lab-main
bash start-lab.sh
```

Open **http://localhost:8765**. For a manual environment setup:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r lab/requirements.txt -r lab/requirements-foundations.txt
.venv/bin/python lab/lab.py doctor
.venv/bin/python server.py
```

Run the server and package installation as your regular user. `sudo` is only used above for the operating-system package manager. If your machine has no graphical desktop, `python3.12 server.py --no-browser` starts the service without trying to open a browser. Keep access private; the [Vast.ai guide](../../course/Vast-ai-Lab.md) explains a private tunnel for a remote host.
