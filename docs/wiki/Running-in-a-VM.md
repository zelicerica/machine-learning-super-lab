# Running the lab in a virtual machine

[Wiki home](Home.md) · [Repository README](../../README.md)

**We recommend a dedicated virtual machine (VM) for the lab, particularly when using the Python workbench or experimenting with packages.** The core CPU exercises should work in a compatible guest with enough memory and disk space. This is a compatibility assessment from the code and package requirements, **not a claim that every hypervisor or guest has been tested**.

## Why use a VM?

Workbench Python runs with the account permissions of the server. A dedicated guest separates that activity from your everyday operating system, documents, and browser profile. A Python virtual environment (`.venv`) only separates package installations; keep using it **inside** the VM.

A VM reduces exposure; it is not a guarantee against malicious code. Keep the hypervisor, host, and guest patched. The core lab does not need your personal credentials or sensitive datasets.

## Recommended starting configuration

These are planning targets, not measured minimums or a promise that every optional model will fit.

| Setting | Starting point |
|---|---|
| Host/guest architecture | x86-64 virtualization on an Intel/AMD host |
| Guest operating system | Ubuntu 24.04 LTS Desktop, 64-bit; install available OS updates |
| Virtual processors | 4 virtual CPUs if the host has capacity; smaller exercises may work with 2 |
| Guest memory | 8 GB assigned to the guest; consider 12–16 GB for larger experiments |
| Guest disk | 40 GB virtual disk as a starting allowance; leave room for the OS, packages, model caches, outputs, and snapshots |
| Host capacity | Leave enough RAM, CPU, and disk for the host; assigning all resources to the guest can make both unusable |
| Python | 64-bit Python 3.12 and its virtual-environment support |
| GPU | None required for the core CPU path |
| Networking | NAT for setup/downloads, with no inbound port forwarding for the lab |
| Browser | Run it **inside the guest**, alongside the Python server |

A supported x86-64 Windows guest is another option; follow [Windows setup](Windows-Setup.md). The project's historical verification covers Linux CPU runs, not a fresh native Windows or VM test matrix.

**Apple silicon and ARM:** the x86-64 VM recipe above does not transfer directly to ARM guests. Native Apple-silicon macOS setup is documented separately, but the pinned environment has not been validated here in Linux ARM or Windows ARM guests. Check wheel availability and architecture before committing to that configuration; do not silently downgrade packages to make installation succeed.

## Set up and use the guest

1. Create the guest with a maintained hypervisor that supports your host and the selected guest. Install and update Ubuntu Desktop from its official image. Use a regular guest account for the lab.
2. Disable shared clipboard, drag-and-drop, shared host folders, and unnecessary device passthrough. Download the repository inside the guest rather than running it from a shared host folder. Avoid signing the guest browser into your everyday synced profile.
3. Use NAT for internet access during package and model downloads. **NAT is not an outbound firewall**: the guest may still reach host services, your network, and the internet. Use firewall restrictions or disconnect its virtual network adapter when those connections are unnecessary.
4. Inside the guest, follow [Linux setup](Linux-Setup.md): install Python 3.12 and `venv`, extract the project, and run `bash start-lab.sh`. Keep the launcher terminal open.
5. In the **guest browser**, open `http://localhost:8765`. Run **Lesson 1 → Practice → Install local packages**, then the diagnostic and [first experiment](First-Experiment.md).
6. Save a clean snapshot after verifying the setup. Before trying package changes, stop the lab and take another snapshot. Back up wanted outputs and exported notes before reverting; later work in the guest will be rolled back too.

Internet is needed for initial packages and pretrained checkpoints. Exercises using only local data can run offline once their dependencies are installed. A pretrained exercise also needs all of its selected model/tokenizer files cached; verify that specific exercise offline before depending on it.

## Keep localhost inside the VM

`localhost` identifies the operating system running the browser. A host browser visiting `http://localhost:8765` reaches the host, not the guest. The simplest setup is to open both the server and browser in the guest. The learner server binds to loopback by default and rejects a non-loopback bind in normal mode.

Do not change the server to `0.0.0.0`, expose port 8765 to your network, or use preview mode to bypass this restriction. Port forwarding alone does not make a service bound only to guest loopback accessible on a guest network address. Advanced remote access is described in the [private-tunnel guide](../../course/Vast-ai-Lab.md); it is not needed for a desktop VM.

## CPU versus optional GPU work

| Workload | VM expectation |
|---|---|
| Lessons and first Python workbench example | No GPU or ML package installation required |
| Foundations, tiny neural builds, retrieval, synthetic data, modern teaching experiments | CPU execution is implemented; actual speed and memory depend on guest resources and experiment settings |
| Default 135M text model and small adapter exercises | CPU path is implemented; allow time and memory for downloads, inference, and training |
| Optional vision inference | CPU code path exists, but uses another downloaded model and additional memory; not a VM-specific benchmark |
| Optional quantized low-rank adaptation (QLoRA) extension | Separate hardware/dependency setup; do not assume a normal CPU VM supports it |

The host having an NVIDIA GPU does not automatically expose CUDA to its guest. Ordinary virtual graphics/3D acceleration is not proof of CUDA access. GPU passthrough or supported virtual GPU configurations require their own hardware, drivers, and validation. They are not required for the recommended CPU track; follow the [optional QLoRA guide](../../course/QLoRA-and-Cloud.md) only when needed.

## Validate your guest before larger runs

- Confirm Python reports `3.12.x`, and the environment diagnostic succeeds. `cuda_available: false` is expected on this CPU setup.
- Run **A first learning loop**, then **Foundations → Linear regression**. Check that a report and artifacts appear; reload the saved model and predict on a new input.
- If using pretrained text, run **Run your first model** and confirm the checkpoint download and inference complete.
- Stop and restart the server; verify the same guest browser at the same address retains progress and that saved artifacts remain available.
- Record the hypervisor, guest OS/architecture, assigned RAM/CPUs, Python/package versions, and tested exercises when reporting compatibility. A successful website launch alone does not verify training or optional models.

## References and implementation evidence

- [Oracle VirtualBox security guidance](https://docs.oracle.com/en/virtualization/virtualbox/7.2/user/Security.html): guest integration, network boundaries, patching, and regular-user operation.
- [Oracle VirtualBox networking](https://docs.oracle.com/en/virtualization/virtualbox/7.2/user/networkingdetails.html): NAT and port-forwarding behavior. Other hypervisors have their own settings.
- [PyTorch installation commands](https://docs.pytorch.org/get-started/previous-versions/): platform-specific CPU packages for the pinned version; availability is not a security endorsement.
- [Ubuntu 24.04 Python virtual-environment package](https://packages.ubuntu.com/noble/python3.12-venv).
- Repository evidence: [server.py](../../server.py), [language-model runtime](../../lab/lab.py), [Foundations runtime](../../lab/foundations.py), [modern teaching runtime](../../lab/modern_lab.py), and [historical verification](../../VERIFICATION.md).
