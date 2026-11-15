# Development and verification

[Wiki home](Home.md) · [Repository README](../../README.md)

`server.py` runs the local web server and jobs. `public/` contains the interface and bundled renderer. `course/` contains lessons and illustrations. `lab/` contains executable model code and data. `tests/` and `lab/tests/` cover the runner and model contracts. `lab/reference-results/` contains clearly labeled measured teaching runs, separate from your own results.

From an environment with the course packages, developer checks are:

```text
python -m unittest discover -s tests -v
python -m unittest discover -s lab/tests -v
python build_course.py
```

Use `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux in place of `python` when the shell does not already select the correct environment. Optional JavaScript syntax check: `node --check public/app.js`. Node.js is not required to run the course.

The previous release passed 39 automated checks and browser training/reload checks on Linux. Windows and macOS installation instructions are documented, but full native model runs on those systems were not executed in that verification. See [VERIFICATION.md](../../VERIFICATION.md) for the latest exact record. Pinned dependencies aid reproducibility and are not a claim to be the newest available versions.

The repository excludes virtual environments, downloaded checkpoints, learner datasets and outputs, logs, credentials, and browser progress. Bundled third-party notices are retained. No project-wide open-source license has been selected; public visibility alone is not a blanket reuse license. Dependencies, downloaded models, and datasets retain their own licenses and usage conditions.

Installation references: [Python on Windows](https://docs.python.org/3.12/using/windows.html), [Python on macOS](https://docs.python.org/3.12/using/mac.html), [Homebrew Python 3.12](https://formulae.brew.sh/formula/python@3.12), [Ubuntu Python 3.12 virtual environments](https://packages.ubuntu.com/noble/python3.12-venv), and [PyTorch 2.6.0 installation commands](https://docs.pytorch.org/get-started/previous-versions/).

## Documentation maintenance

The short README links to the GitHub Wiki. These Markdown sources in `docs/wiki/` also ship with ZIP downloads. Keep the source pages and published wiki synchronized. When publishing to GitHub Wiki, replace local `Page.md` links with wiki page URLs and `../../` repository links with absolute repository URLs. Do not move the course workbooks or generated lesson content when reorganizing documentation.

This documentation revision is based on source inspection of the CPU execution paths and published virtualization guidance. It does **not** add a hypervisor or guest-OS test result to the historical verification record. VM compatibility is expected under the stated conditions; use the guest validation checklist before relying on an installation.
