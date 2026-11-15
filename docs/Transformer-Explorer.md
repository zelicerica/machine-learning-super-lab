# Transformer explorer — implementation and verification

Release: `2026.11-transformer.1` · Lesson 59 · Route: `#lesson/transformer-world`

The new concept lesson pairs the original encoder–decoder architecture with a numerical attention activity and a research timeline. Seven annotated primary/reference sources accompany the lesson. Existing lesson IDs and numbers 1–58 are preserved.

## Implementation

- `public/transformer.js` projects 3D box vertices through an orbit camera onto a 2D canvas. Faces are depth-sorted; pointer selection uses the projected face polygons. This is a lightweight software-rendered 3D schematic, not a downloadable mesh or trained network visualization.
- Mouse, touch, keyboard, zoom buttons, component selection, reset/front views, spacing, and label controls operate locally. The renderer schedules frames only when needed and releases listeners, animation frames, and resize observation on navigation. There is no automatic rotation.
- A native component menu and textual descriptions remain available when canvas is unavailable. The attention calculator also works without canvas. Narrow views use shorter canvas labels; the inspector retains complete names and explanations.
- The original model has six encoder layers and six decoder layers. The scene shows one layer's structure from each stack, explicitly labeled as repeated, rather than pretending that the diagram contains all twelve layers. The cross-stack arrow denotes final encoder memory. Residual connections are explained textually.
- No runtime dependency, external script, CDN, model download, or remote API is added. No model training code or server access control changes are part of this feature.

HeyGen was requested for a separate eight-second conceptual introduction. That service produces video rather than an interactive mesh. Its player is delivered separately in the development conversation; the application does not embed a pending asset or depend on that service. The interactive schematic itself is code-native. A future bundled video must be reviewed and clearly labeled as conceptual rather than a literal architecture or observed model activation.

## Validation — November 15, 2026

| Check | Result |
|---|---|
| `python build_course.py` | 59 lessons and 17 reference chapters |
| `node tests/check_course_dom.cjs` | 76 lesson/reference routes and 69 Mermaid diagrams; existing saved progress retained |
| `node tests/check_transformer.cjs` | Numerical reference cases, perspective projection, orbit/pan/zoom, touch gesture simulation, keyboard, controls, fallback, and lifecycle cleanup passed |
| `python -m unittest discover -s tests -p test_server.py -v` | 16 tests passed, including origin/token/host and traversal protections |
| `python -m unittest discover -s tests -p test_content_audit.py -v` | 8 tests passed in the existing test environment |

The Node checks need the existing developer `jsdom` package; they do not add it to the learner's runtime. `SML_JSDOM_PATH` can point to its installed directory. The content audit uses the existing machine-learning test environment.

The canvas tests simulate drawing and input events. **Browser layout, visual rendering, and physical touch-device behavior were not verified in this session:** both browser environments could not reach the isolated localhost server. A standalone data-URL preview was rejected by browser URL policy. No workaround was used to bypass that restriction. Full training, model inference, and VM execution were not rerun for this front-end/content change.

To finish a manual visual review, run the lab normally in the same operating system as your browser, open Lesson 59, and exercise orbit, pan, wheel zoom, component selection, both themes, and a narrow window. Confirm the attention output changes from `[1.6044, 1.1978]` to `[1.3395, 0.6605]` when the causal mask is enabled with the initial query. Check that navigating away and back creates only one viewer.
