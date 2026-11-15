# Developer reference evidence

**Content audit, midnight.4:** GAN reports use fresh samples from the same simulator; grid-world RL evaluates the same map and start. Neither has a reserved final-test split. Misleading historical split labels have been corrected without changing measurements. Midnight.4 also gives GAN evaluation its own random generator, so older GAN metrics are not exact expected outputs of the revised script.

These measured reports and plots come from the development verification runs, not the learner’s computer. All 18 baseline builds and 18 changed-setting runs completed, saved artifacts, and reloaded their models. Use the same seed and documented settings when comparing. Runtime varies by hardware. Production performance is not established by these teaching tasks. Model weights and executable serialized objects are deliberately excluded from this reference folder; create your own through the lessons.
