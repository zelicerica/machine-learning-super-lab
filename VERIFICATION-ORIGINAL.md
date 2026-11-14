# Verification scope

Checked in the development environment with Python 3.12.14 on Linux and a desktop Chrome browser.

## Website

- Eight server tests passed: real code execution and saved logs; actual exception output; stopping a running process; cross-origin request rejection; missing session-token rejection; untrusted Host rejection; path traversal and unknown-source rejection; terminal command and output-path validation; and a regression check separating tokenizer vocabulary files from evaluation reports. Some tests cover more than one assertion.
- JavaScript syntax check passed.
- Browser editor ran the numerical learning loop and displayed the actual five-step loss sequence.
- Browser terminal trained the tiny Transformer for 20 steps: validation loss decreased from 5.705249 to 2.735432. This is a synthetic teaching corpus, not real-world language proficiency.
- Browser terminal fine-tuned the real SmolLM2 checkpoint for five steps, with 460,800 trainable adapter parameters. Validation loss changed from 2.112239 to 0.323754.
- Browser evaluation of that five-step adapter completed on 12 validation examples: accuracy 7/12 (58.3%), valid output structure 11/12 (91.7%). The results page displayed these actual scores correctly. This brief run is a workflow test, not the course's recommended 60-step assessment run or a production-quality claim.
- The loss chart displayed actual training history. Tokenizer and configuration files are excluded from experiment reports.
- Incorrect comprehension answers kept completion disabled; a correct answer enabled self-reported completion. Progress survived page reload.
- Model-family diagram loaded; desktop layout, navigation, editor, terminal, results, and assessment controls were inspected.

## Earlier machine learning exercise checks

See `lab/VERIFICATION.md` and `lab/reference-results/` for the original course's eight model tests, 60-step adaptation, inference, retrieval, vision smoke test, and toy architecture runs. These are reference evidence, not the learner's own experiment results.

## Limits

Windows launch scripts and PowerShell instructions were reviewed but could not be executed on Windows in this environment. A user's screenshot confirmed their Python 3.12.10 executable worked; it did not verify this entire website on that machine. macOS, narrow-screen browser interaction, graphics processor acceleration, and Vast.ai rental were not tested here. Browser image upload and the one-click package installer were not exercised end to end in this website test; the underlying vision script and equivalent package environment were exercised separately.

The downloaded project excludes development environments, model weights, browser test output, and test-session progress. Learners begin with their own environment and reports. The interface makes no external script or font requests. Model and package downloads occur only when requested by the corresponding actions.
