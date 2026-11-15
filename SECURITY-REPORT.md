# Security test candidate results — 2026-11-15

GitHub release **2026.11-transformer.2** preserves upstream Lesson 59 and integrates the owner-tested setup-fix package. Combined validation passed **62 Python course/server/setup tests**, **76 DOM routes / 69 diagram parses**, and the dedicated Transformer numerical/interaction/lifecycle simulation checks. See `VERIFICATION.md` and `security/test-evidence/github-integrated-*.log`. The earlier ZIP results below retain their original 58-lesson scope; no unrun model or visual checks are promoted to passes.

**Suitable for isolated functionality testing; not recommended as an in-place upgrade to the working lab yet.** Read `SECURITY-TESTING.md` before installing. This source-only package deliberately excludes environments, learner inputs/outputs/runs, model weights, private configuration and caches.

## Browser setup correction

Release **2026.11-midnight.4-security-test.3** removes the 50-character folder restriction introduced in the earlier candidate. That restriction incorrectly rejected ordinary Downloads extraction paths before installation began. Browser setup now creates a fresh `.lab-environments/env-...` environment with Windows extended-length paths, uses Python's bundled pip only to bootstrap installation of the exact hashed candidate wheels, removes the optional Telegram helper, checks dependency consistency and real learning-library imports, and selects the runtime only after those checks. Existing environments remain untouched; a previous `.runtime-python` configuration is backed up before replacement. Setup and validation subprocesses have timeouts. No administrator/security-policy change is made.

An actual **HTTP POST `/api/run` setup job on localhost:8766** completed in a fresh long-path fixture using all **57 same pinned wheel hashes**, with an existing `.venv/keep.txt` preserved. `ready` and `foundations_ready` were both true, exactly one fresh managed environment was created, and the Telegram helper was absent. This test used a local verified wheelhouse with indexes disabled; only the CPU requirement's source was changed from its official URL to the equivalent exact local version for the offline fixture. No new online download was made. Logs and result: `security/test-evidence/browser-setup-long-path.log` and `browser-setup-long-path-result.json`. **62 course/server/setup tests passed**, including runtime-config backup and existing-environment preservation (`browser-setup-regression.log`). A full online download from the user's browser and an OS/browser security dialog were not observed; an unexplained security warning still needs its exact text and must not be bypassed.

## Original-brightness video revision

Release **2026.11-midnight.4-security-test.2** adds the requested **Dim video / Original video** top-bar control. Original mode removes the CSS gradient overlays and sets video opacity to 1 across routes/themes/screen-width rules; dimmed mode retains the prior reading treatment. The preference has its own local-storage key and does not change motion/reduced-motion settings. No video transcoding or replacement occurred: SHA-256 remains `4ff4b64b76644f0aa576dc5d52c5d9fbbf7e9d1f00259a90044d219e19cf32d7`.

The updated simulated DOM check passed **75 routes / 69 diagram parses**, including default original brightness, toggle button state, stored choice, navigation and playback independence (`security/test-evidence/background-tone-dom.log`). Browser preview access was unavailable, so no new visual layout/video playback review is claimed. The temporary test server was stopped. The following Python/HTTP/model results describe the preceding security candidate; backend and dependency versions are unchanged in this brightness revision.

## Verified results

| Check | Actual result | Evidence under `security/test-evidence/` |
|---|---|---|
| Candidate Windows CPU package imports | PyTorch 2.13.0+cpu, Transformers 5.23.0, PEFT 0.21.3, Accelerate 1.15.0, Pillow 13.0.0 import | Offline adapter report also records versions |
| Dependency resolution | No broken requirements found | `candidate-final-pip-check.log` |
| Course/server/foundation/modern regressions | 60 passed | `candidate-regression-2.log` |
| Original model-mechanism tests | 8 passed | `candidate-original-model-tests.log` |
| Tiny random-model Low-Rank Adaptation (LoRA) | 4 optimizer updates; frozen base unchanged; saved/reloaded logits identical; new-input generation completed | `offline-lora-report.json`, contract, log |
| Independent synthetic-token evaluation | Loss 4.191212 before, 4.195366 after | Same report; this is a compatibility exercise, not a quality improvement |
| Actual loopback HTTP flow on 8766 | Workbench, Foundations training, example loading, model reload and fresh-input prediction completed; predictions 38.999999999999886 and 58.999999999999915 | `candidate-http-flow.json`, `candidate-http-check-2.log` |
| Interface simulated Document Object Model (DOM) | 75 routes, 69 Mermaid parses passed; no visual browser assertion | `dom-candidate-final.log` |
| Requested optional Telegram-helper removal | Core tqdm progress iteration works; helper absent in the existing live runtime and review environments | `telegram-removal-smoke.log`; new package setup runs its own removal command |

The first full course regression attempt failed two UTF-8 reads under Windows defaults; explicit encoding fixes led to 60/60 passing. The first HTTP harness failed while parsing trailing job-output text after successful jobs; its JSON parser was corrected, and a fresh-output rerun passed. Both failed and successful logs are retained. These harness failures are not hidden as product successes. The adapter smoke artifacts remain in the review workspace; only their report, contract and recorded hashes are shipped. Rerunning the included command creates fresh artifacts in your test copy.

## Findings and remediation

| Finding | Assessed severity and exposure | Candidate remediation |
|---|---|---|
| Public preview binding and bypassed Host checks | High if launched with the old npm development command; original live listener was private | All modes loopback; preview execution rejected; exact Host/Origin and cross-site checks |
| Existing environment could be reused by full setup | Medium preservation/reliability risk | Create a fresh managed environment; preserve existing environments/config backup and use hashed Windows artifact lock |
| Windows cancellation ignored taskkill failure | Medium reliability; arbitrary user workbench subprocess trees remain an OS trust boundary | Check exit status and terminate worker on failure |
| Known package advisories | High/critical-class records exist in original dependency families; exact reachability depends on input/model APIs | Candidate upgrades listed below; original advisory records and candidate scan snapshot included |
| Executable model formats and workbench | High impact for untrusted code/artifacts; intentional local-user capability | Retain trusted-artifact boundary, safetensors/weights-only restrictions already present; document residual risk rather than claim an OS sandbox |

The original installed versions flagged by advisory queries were Accelerate 1.10.1, Pillow 11.3.0, pip 25.0.1, setuptools 78.1.0, PyTorch 2.6.0+cpu and Transformers 4.57.1. Representative source records include [Accelerate checkpoint traversal](https://github.com/advisories/GHSA-4j2p-28q2-5m79), [Transformers model loading](https://github.com/advisories/GHSA-29pf-2h5f-8g72), [PyTorch untrusted checkpoint handling](https://github.com/advisories/GHSA-63cw-57p8-fm3p), and [setuptools download traversal](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf). Raw saved affected ranges, severity fields, aliases and references are in `security/baseline-advisory-records.json`; advisory aliases must not be counted as separate issues. Some records concern APIs not used by the core exercises; no exploit against the live lab was attempted.

Candidate versions: torch **2.13.0+cpu**, transformers **5.23.0**, peft **0.21.3**, accelerate **1.15.0**, Pillow **13.0.0**, pip **26.3**, setuptools **84.2.1**, Mermaid **12.3.0**. NumPy, scikit-learn, SciPy and Matplotlib direct pins retained. All **57 candidate wheel artifacts** matched official registry digests and had no non-withdrawn OSV records in the saved query snapshot. The separate npm tooling audit reported zero advisories at query time. These are dated scan results, not an infostealer clearance or a promise of future security.

## Package provenance and malicious-package review are separate

The original live environment contained 43 distributions. Before the explicitly requested helper removal, 42 non-PyTorch exact-version packages matched official registry wheel hashes and **11,402 installed package files** matched those wheels. PyTorch's locally recorded hashes matched, but its original canonical binary wheel was not independently downloaded and compared. No unexplained file mismatch was found in those checks.

Static triage inspected wheel zip contents, declared dependencies, startup/build hooks, native-file lists, subprocess/environment and credential/network indicators without executing suspicious artifacts. Observed Telegram indicators came from tqdm's optional opt-in progress integration, not a discovered infostealer. Its module was removed at the owner's request; this was not remediation of proven malware. No exhaustive native-code analysis, full outbound traffic trace or guarantee against compromised upstream maintainers is claimed.

Software Bills of Materials (SBOMs), in CycloneDX format, describe original and candidate Python components, versions, dependency declarations and available artifact provenance. `security/javascript-inventory.json` identifies exact embedded version labels where available; the original and replacement prebuilt bundles do not disclose enough evidence here to certify every embedded transitive version. `review-tooling-package-lock.json` inventories review tooling and must not be mistaken for the prebuilt bundle's upstream build lock. There is no need to install Node/npm tooling to run the lab.

## Not verified; keep as test candidate

- Actual pretrained SmolLM2 inference, its ticket-data LoRA training/reload and independent benchmark evaluation on these new versions.
- Actual SmolVLM checkpoint/image inference, optional Qwen checkpoint and original learner saved-artifact cross-version compatibility.
- GPU execution, GPU package variants, native macOS/Linux setup, visual browser rendering/video playback, and an exact fresh online installation after extracting this ZIP.
- Python/OpenSSL/OS security patching. The review machine used Python 3.12.10 with OpenSSL 3.0.16; this package changes neither.
- Full malicious-package clearance or all original JavaScript transitive build provenance.

These limits are why the package is safe to approach as an isolated test copy rather than a blanket “safe to apply” release. Follow the manual test checklist and preserve actual outputs before any future switch.

## Patch and archive inspection

`security/source-changes.patch` provides textual source changes. The large bundled renderer replacement is supplied as a full file, with before/after SHA-256 digests in `security/changed-files.json` and `javascript-inventory.json`. Added security reports/SBOMs/evidence are full files. `security/package-files-sha256.json` covers ZIP contents except itself. The accompanying external `.sha256` file covers the ZIP archive. Backup, future promotion, restart and rollback commands are in `SECURITY-TESTING.md`; none were executed against the live installation.
