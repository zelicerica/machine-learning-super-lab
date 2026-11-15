# Getting started

[Wiki home](Home.md) · [Repository README](../../README.md)

## Choose your environment

**Recommended: run the lab in a dedicated virtual machine (VM).** Follow [Running in a VM](Running-in-a-VM.md) for the CPU path, isolation settings, and validation checklist. A Python `.venv` isolates packages; it does not isolate the operating system.

## What runs where

Your browser displays the lessons, editor, and results. A Python server on **your computer** runs the exercises and saves their outputs. This is a real Python workbench, not an isolated operating-system emulator.

- The server and the first editor example need only Python's standard library.
- Model exercises need the pinned dependencies installed once in a project virtual environment.
- Many foundation and synthetic-data labs need no model download. Pretrained exercises download their selected checkpoint on first use.
- The starting language model is SmolLM2 with 135 million parameters. Core teaching exercises use a **central processing unit (CPU)**; a **graphics processing unit (GPU)** and a paid cloud account are optional.
- No Git, Node.js, Git Bash, Docker, or application programming interface key is required for the normal download-and-run path.
- The website binds to localhost by default. Keep it local: workbench code runs with your operating-system account's permissions.

The Python backend is required. Opening `public/index.html` directly or enabling GitHub Pages will not run the Python exercises.

## Get the files

On the repository page, choose **Code → Download ZIP**, extract the entire archive, and open the extracted folder containing `server.py` and `start-lab.cmd`. Do not run from inside the compressed archive. Your folder may be named `machine-learning-super-lab-main`; use its actual location in the examples below.

If Git is already installed, cloning is an alternative:

```text
git clone https://github.com/zelicerica/machine-learning-super-lab.git
cd machine-learning-super-lab
```

Use **64-bit Python 3.12** for this pinned teaching environment. Later 3.12 patch releases are suitable; another minor version has not been validated here. Prefer a current supported 3.12 package from your platform's package manager. Package installation and model downloads require internet access. Allow several gigabytes of free disk space for dependencies, caches, and experiments; requirements vary by selected model. A computer with 8 gigabytes of memory is a reasonable starting target for the tiny exercises, not a guarantee for every optional model.

## Install and launch

Choose [Windows](Windows-Setup.md), [macOS](macOS-Setup.md), or [Linux](Linux-Setup.md), then [run your first experiment](First-Experiment.md). In a VM, follow the instructions for the **guest** operating system, and keep both the server and browser inside that guest.
