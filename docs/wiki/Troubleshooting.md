# Troubleshooting

[Wiki home](Home.md) · [Repository README](../../README.md)

| Symptom | Next step |
|---|---|
| `py` missing or `python` opens the Microsoft Store | Use `start-lab.cmd` or the direct executable in the Windows section |
| `python3.12` not found | Install Python 3.12, reopen Terminal, and check the versioned command; see your platform section |
| `No module named venv` / `ensurepip` failure on Linux | Install your distribution's Python 3.12 virtual-environment package |
| `No matching distribution` on macOS | Check Apple silicon versus Intel and native Python architecture; see the Intel limitation |
| Missing machine-learning package | Install in the same `.venv` used by the website; run the diagnostic again |
| Port 8765 in use | Check for another running lab. Stop it, or run the server with `--port 8766` and use that port |
| Model download fails | Save the complete error; check connectivity, disk space, and the pinned checkpoint. Cached files must be complete for offline use |
| Slow generation or out of memory | Start with the small default checkpoint and shorter output. Use cloud hardware only after measuring a need |
| Failed or interrupted run | Read the terminal's actual error, retain its log, and use a new output directory for the next attempt |

For a useful bug report include your operating system and architecture, Python version, lesson number, exact command, diagnostic output, and full error text. Remove credentials and private data before sharing.

## VM-specific checks

| Symptom | Next step |
|---|---|
| Host browser cannot reach the lab | Open the browser **inside the VM**. Guest localhost and host localhost are different. No port forwarding is needed for this path. |
| VM reports `cuda_available: false` | Expected for the CPU track. A virtual display adapter is not a CUDA GPU. |
| Guest becomes slow or runs out of memory | Check guest RAM, host memory pressure, free disk space, and assigned virtual CPUs; reduce workload size before adding optional models. |
| Package installation fails on an ARM guest | The recommended VM recipe targets x86-64. Do not assume the pinned packages have matching ARM wheels. |
| Existing progress seems missing in a VM | Browser progress belongs to the original browser profile and website origin. Import exported notes where supported; moving source files does not migrate browser storage. |

See [Running in a VM](Running-in-a-VM.md) for the complete recommended configuration.
