# Quantized adaptation and a bounded cloud experiment

**Tag: Concept / optional hardware-specific extension.** All core modern-track experiments run on a central processing unit. `lab/qlora_gpu.py` contains a genuine NormalFloat4 loading/adaptation path, but its graphics-processor execution was not tested in the CPU release environment. Do not interpret source compilation or dependency installation as proof of a completed graphics-processor run.

The local `packed4` branch is intentionally different: it uses uniform quantization on selected projections, packs two codes into each byte, and unpacks to floating point for computation. The optional extension uses a compatible bitsandbytes backend, a four-bit NormalFloat quantizer, double quantization of constants, and trainable low-rank adapters. It uses ordinary AdamW rather than a paged optimizer, so it does not claim to reproduce every QLoRA paper mechanism.

## Decide before renting

1. Finish the local baseline. Record model size, sequence length, batch size, training time and memory.
2. Decide whether the goal requires this backend or a larger experiment. A cloud instance is not necessary simply to understand quantization.
3. Choose a hardware/runtime combination supported by the current [bitsandbytes installation guidance](https://huggingface.co/docs/bitsandbytes/main/en/installation) and [quantized adaptation guidance](https://huggingface.co/docs/peft/developer_guides/quantization).
4. Compare live offers in Vast.ai. Price, availability, reliability, disk allocation and transfer charges vary. This guide supplies no invented hourly quote.
5. Set an affordable maximum runtime and spending budget. A long-context or multi-model reinforcement-learning job needs a different estimate than this tiny adapter exercise.

| Budget component | Calculation or check |
|---|---|
| Compute | Observed hourly rate × planned hours, including setup and idle time |
| Storage | Allocated space × retained duration × applicable storage rate |
| Transfer | Applicable ingress/egress charges from the selected offer |
| Headroom | Failed runs, downloads, checkpoint saves and startup overhead |
| Memory | Base tensors + quantization metadata + adapters + gradients + optimizer + activations + temporary buffers |

**Worked cost exercise:** if your chosen offer shows rate R and you plan two hours, compute cost is 2R. Add the actual storage and transfer terms. R is a learner-entered rate, not a quoted market price. Write down what you will do if the first measured run takes twice the expected time.

## Connect from Windows, macOS, or Linux

Use Vast.ai's current connection instructions and the host/port shown for your own instance. Prefer a maintained PyTorch template with the intended CUDA support already installed. Inspect the template and provider controls before placing data on it. Use original public teaching data for this exercise.

From **Windows PowerShell or Command Prompt**, use the built-in OpenSSH client if installed. Replace the example host and port with the values Vast.ai provides; do not type the placeholders literally.

```text
ssh -p 12345 root@YOUR_INSTANCE_HOST
```

The same command works from a macOS or Linux terminal. `root` must match the account provided by the selected template. No Git Bash is required on Windows.

**Remote Linux shell only:**

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
git clone https://github.com/zelicerica/machine-learning-super-lab.git
cd machine-learning-super-lab
python -m venv --system-site-packages .venv-cloud
.venv-cloud/bin/python -m pip install -r lab/requirements.txt -r lab/requirements-foundations.txt
.venv-cloud/bin/python -m pip install bitsandbytes
.venv-cloud/bin/python -m pip check
.venv-cloud/bin/python -m pip freeze > cloud-environment.txt
```

This isolated environment can reuse the template's graphics-enabled PyTorch; inspect `pip check` and the versions before training. Do not install the laptop's processor-only PyTorch wheel over the cloud image. The bitsandbytes version is intentionally selected against current hardware support, then recorded in the environment file; this is not a pinned universally compatible cloud environment.

```bash
cd lab
../.venv-cloud/bin/python lab.py doctor
../.venv-cloud/bin/python lab.py lock
../.venv-cloud/bin/python qlora_gpu.py --steps 5 --output outputs/nf4-smoke-01
../.venv-cloud/bin/python qlora_gpu.py --steps 30 --output outputs/nf4-train-01
```

Five updates check the workflow. Thirty do not establish model quality. Inspect finite losses, validation behavior, actual memory and the saved/reloaded logit comparison. The adapter must be reloaded against the same pinned base revision. Use the existing ticket benchmark only as a teaching check; broader independent data are required for useful deployment.

The script fails clearly when no compatible CUDA device exists. Hardware-specific package problems remain possible; consult the exact backend version's documentation rather than silently switching quantizers and reporting the original experiment name.

## Optional private browser connection

To run the course website on the remote instance, keep it bound to loopback. From the remote project root:

```bash
.venv-cloud/bin/python server.py --no-browser
```

From a second **local PowerShell, Command Prompt, macOS, or Linux terminal**:

```text
ssh -p 12345 -L 8765:127.0.0.1:8765 root@YOUR_INSTANCE_HOST
```

Then open `http://localhost:8765` locally. If that local port is occupied, use a different left-hand port in the forwarding command. The browser runner's new small mechanisms intentionally use the central processing unit; the optional graphics training command above runs from the remote shell.

## Preserve evidence and end charges deliberately

Copy the adapter folder, reports, `model-lock.json`, environment file and commands back before deleting the instance. For example, from the **local terminal**, use the real remote path shown by `pwd`:

```text
scp -P 12345 -r root@YOUR_INSTANCE_HOST:/ACTUAL_PROJECT_PATH/lab/outputs/nf4-train-01 ./nf4-train-01
```

Verify that the downloaded files open and retain model/base identity. Stopping a compute instance can leave storage allocated and billable. Destroying an instance can delete its local disk. Review the selected offer and current [Vast.ai instance-management documentation](https://docs.vast.ai/guides/instances/manage-instances); inspect the dashboard for remaining instances, volumes and charges after cleanup.

**Assessment:** submit local baseline measurements, the rental decision, actual hardware/package versions, a spend calculation, a verified downloaded checkpoint, reload evidence, and the cleanup record. Record failures and any unexecuted steps explicitly.
