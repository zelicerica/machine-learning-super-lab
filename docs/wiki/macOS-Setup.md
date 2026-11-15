# macOS — Apple silicon

[Wiki home](Home.md) · [Repository README](../../README.md)

Start with [Getting started](Getting-Started.md) to download the project and check requirements. Prefer a [dedicated VM](Running-in-a-VM.md) when its guest platform supports the pinned packages.

These instructions target an Apple silicon Mac using a native `arm64` Python. Check **Apple menu → About This Mac** for the chip, or run `uname -m` in Terminal. If an Apple silicon Mac reports `x86_64`, use a native Terminal and Python rather than running through Rosetta.

### 1. Install Python 3.12

If you already use [Homebrew](https://brew.sh), run:

```bash
brew install python@3.12
python3.12 --version
```

If Homebrew is not installed, follow its official installation instructions first, or use the [official Python macOS installation guidance](https://docs.python.org/3.12/using/mac.html). Do not replace Apple's system Python. With Homebrew, `python3.12` is the versioned executable; an unrelated `python3` may refer to another version.

### 2. Launch

In Terminal, change into the extracted folder, using its actual path:

```bash
cd ~/Downloads/machine-learning-super-lab-main
bash start-lab.sh
```

Open **http://localhost:8765** and continue with [your first experiment](First-Experiment.md). Using `bash start-lab.sh` works even if extraction did not preserve executable permissions.

### Optional manual dependency installation

From the project folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0
.venv/bin/python -m pip install -r lab/requirements.txt -r lab/requirements-foundations.txt
.venv/bin/python lab/lab.py doctor
.venv/bin/python server.py
```

The macOS PyTorch command deliberately uses the normal package index, not the Linux/Windows CPU index. The default exercises remain on the CPU. Metal Performance Shaders acceleration is not a verified course configuration, and NVIDIA CUDA availability is expected to be false on a Mac.

**Intel Mac limitation:** the pinned PyTorch 2.6.0 environment does not have official macOS Intel wheels. The standard-library website and first workbench example can still run, but the full neural-model environment needs a compatible Linux/Windows machine or a separately validated environment. Do not silently downgrade PyTorch and assume the remaining pinned dependencies still work. See the [official Intel macOS build deprecation](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
