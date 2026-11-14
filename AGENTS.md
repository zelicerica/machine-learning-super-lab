# Machine Learning Super Lab teaching standard

Teach new engineers from intuition through algorithms to inspectable implementations. Expand abbreviations on first use. Use PowerShell and Command Prompt for local Windows instructions; label remote Linux commands explicitly.

Every model-family section must include a real-world use case and a **Build from start to finish** section: problem and success criterion; permitted data and feature/target contract; preprocessing and split; architecture; training or adaptation; baseline and independent evaluation; model saving and reloading; inference on a new input; local deployment; production gaps; monitoring and rollback; cost plan; and assessable evidence.

Tag material **Concept** or **Runnable implementation lab**. A concept includes a design activity and explanation check. A runnable tag requires included executable code and a repeatable command. State whether the scope is a small mechanism demonstration, pretrained inference, adaptation, or training from random initialization. Do not present plans as implemented features or teaching data scores as production results.

Prefer small local workloads. Recommend Vast.ai only when measured memory or runtime justifies it. Explain connection, hardware verification, price components, backup, stop versus destroy, and retained storage. Keep localhost code execution private. Do not rent paid resources automatically.

Include numerical examples, algorithms, diagrams, architecture flows, visual explanations, failure analysis, and change-one-variable experiments. Preserve independent evaluation and actual outputs. Each implementation should save a reproducible report, model/input contract, and artifacts. Verify model reload and inference. Do not overwrite a learner's existing outputs, virtual environment, notes, or progress during upgrades.
