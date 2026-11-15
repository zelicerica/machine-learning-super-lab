# Machine Learning Super Lab

**Understand it. Run it. Prove it.** Learn machine learning through visual explanations, executable Python experiments, and evidence-based assessments.

**58 lessons · 17 reference chapters · 18 complete Foundations builds**

Explore classical models, neural networks, language models, retrieval, synthetic data, and modern language engineering. Tiny models and teaching datasets help you inspect the process; their scores are not production benchmarks.

**[Open the wiki](https://github.com/zelicerica/machine-learning-super-lab/wiki)** for setup, learning paths, troubleshooting, upgrades, and developer documentation. [Browse the bundled wiki](docs/wiki/Home.md) when reading a downloaded copy.

## Recommended: run in a virtual machine

Use a **dedicated virtual machine (VM)** to separate package installation and workbench code from your everyday computer. The core lab has a CPU execution path and should run in a compatible VM with enough resources; a GPU is optional. **VM-specific execution has not been verified here.**

A practical starting point is **Ubuntu 24.04 Desktop, x86-64, 4 virtual CPUs, 8 GB guest RAM, and a 40 GB virtual disk**, while leaving resources for the host. These are starting targets, not guarantees for every model. Keep the server and browser **inside the VM**, use NAT without lab port forwarding, and disable shared host folders and clipboard integration. See [Running in a VM](docs/wiki/Running-in-a-VM.md) for setup, limitations, and validation.

A Python `.venv` isolates packages, not the operating system. The workbench executes real Python with your account's permissions; keep the service on localhost. VM isolation does not replace package review or security updates.

## Quick start

1. Choose **Code → Download ZIP** on this repository and extract it completely. Open the folder containing `server.py`.
2. Install **64-bit Python 3.12** in your chosen environment. Follow [Windows](docs/wiki/Windows-Setup.md), [macOS](docs/wiki/macOS-Setup.md), or [Linux](docs/wiki/Linux-Setup.md) setup. For a VM, use the guest operating system's instructions.
3. Launch from that folder:

   | Operating system | Launch |
   |---|---|
   | Windows | Double-click `start-lab.cmd`, or run `.\start-lab.cmd` in PowerShell |
   | macOS / Linux | Run `bash start-lab.sh` in Terminal |

4. Open **http://localhost:8765** in the same operating system as the server; in a VM, use the **guest browser**. Keep the launcher terminal open.
5. Choose **Lesson 1 → Practice → Install local packages**, then run the diagnostic and [first experiment](docs/wiki/First-Experiment.md). `cuda_available: false` is normal for CPU use. On an existing GPU environment, follow the wiki before reinstalling packages.

The website and first workbench example need only Python's standard library. Model exercises need additional packages; pretrained exercises also download model weights. No paid cloud account, API key, Git, Node.js, or Docker is required for the normal download-and-run path. Stop the server with **Control+C**.

## Find your next step

| Goal | Documentation |
|---|---|
| Learn the prerequisite concepts | [Tiny AI — AI for Curious Minds](https://from-number-boxes-to-tiny-ai.ericazelic.chatgpt.site) |
| Choose a track or workbook | [Learning paths](docs/wiki/Learning-Paths.md) |
| Fix setup or runtime problems | [Troubleshooting](docs/wiki/Troubleshooting.md) |
| Upgrade and preserve saved work | [Updates and backups](docs/wiki/Updates-and-Backups.md) |
| See what changed | [Release notes](docs/wiki/Release-Notes.md) |
| Understand the code and checks | [Development and verification](docs/wiki/Development-and-Verification.md) |
| Inspect actual test evidence | [Verification record](VERIFICATION.md) |

Public visibility is not a blanket reuse license: no project-wide open-source license has been selected. Dependencies, models, and datasets retain their own terms.
