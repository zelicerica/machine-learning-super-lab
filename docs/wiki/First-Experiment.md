# Run your first experiment

[Wiki home](Home.md) · [Repository README](../../README.md)

Complete [setup](Getting-Started.md) and open `http://localhost:8765` in the same operating system that runs the server. For a VM, use the browser inside the guest.

1. Open **Lesson 1 → Practice → Install local packages**, unless you already completed a manual installation. Wait for `SUCCEEDED` in the website's terminal. Fresh setup installs the course CPU environment; on an existing cloud/GPU environment, use **Add foundation packages** instead of replacing its PyTorch build.
2. Run the diagnostic. A false `cuda_available` result is normal for the default CPU path; CUDA refers to NVIDIA's GPU computing platform.
3. Try **Python workbench → A first learning loop**. This example also works before package installation.
4. Select **Foundations → Linear regression** for an approachable complete build, or **Language models → Run your first model** for the first pretrained checkpoint download.
5. Predict the result before running. Inspect actual metrics and plots in **Experiment results**, then load a saved example, change an input, and predict without retraining where that interface is provided.
6. Complete the comprehension check and write evidence notes. A completed lesson is your record of practice, not a professional certification.

For a direct synthetic-data run, keep the working directory at `lab`, where the course commands expect to run:

```powershell
# Windows PowerShell, starting in the website folder
Push-Location .\lab
& ..\.venv\Scripts\python.exe .\synthetic_data.py --output outputs\my-first-synthetic-run
Pop-Location
```

```bash
# macOS / Linux, starting in the website folder
cd lab
../.venv/bin/python synthetic_data.py --output outputs/my-first-synthetic-run
cd ..
```

Use a fresh output name for each experiment. Stop the website with **Control+C** in its launcher terminal. Use the website's **Stop run** button to cancel an experiment; an interrupted run can leave partial output.
