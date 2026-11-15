# Updates and backups

[Wiki home](Home.md) · [Repository README](../../README.md)

Close the server and back up the project before updating. For a downloaded archive, extract the new version separately, then merge source files into your existing project. Preserve `.venv`, `lab/outputs`, `lab/inputs`, and `runs`. If you edited course source, keep your changes before replacing files. For a Git checkout, review local changes before pulling updates.

This expansion preserves all existing lesson identifiers. It adds Lessons 43–58 and the **Modern language engineering** overview filter. Open **Landmark research** in the sidebar for the new references page. No new package family is required for the core CPU track beyond the existing course environment. The optional GPU QLoRA extension has separate dependencies and is explicitly unvalidated on GPU hardware.

Progress, evidence notes, and editor drafts are stored in your browser profile for the exact website address. Continue using `http://localhost:8765`; a different port, browser profile, or `127.0.0.1` address has separate browser storage. Export notes and terminal logs for portable backups. Trained artifacts and reports remain in the project folder.

An existing environment can be reused through `SML_LAB_PYTHON`. Set it in the same terminal that launches the server. PowerShell: `$env:SML_LAB_PYTHON = "C:\actual\environment\Scripts\python.exe"`; macOS/Linux: `export SML_LAB_PYTHON="/actual/environment/bin/python"`. Use the actual executable, not a guessed path.

## VM backups

Stop the lab and take a VM snapshot before changing packages. Export notes and back up wanted experiment outputs separately before reverting: a snapshot restore also rolls back guest files and browser progress created afterward. Recreate `.venv` inside a new guest rather than copying a host environment across operating systems.

For folder migration and old launchers, follow [Upgrade to Midnight](../../UPGRADE.md).
