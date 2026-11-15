# Transformer explorer — implementation and verification

Release: `2026.11-transformer.3` · Lesson 59 · Route: `#lesson/transformer-world`

The new concept lesson pairs the original encoder–decoder architecture with a numerical attention activity and a research timeline. Seven annotated primary/reference sources accompany the lesson. Existing lesson IDs and numbers 1–58 are preserved.

## Implementation

- `public/transformer.js` projects 3D box vertices through an orbit camera onto a 2D canvas. Faces are depth-sorted; pointer selection uses the projected face polygons. This is a lightweight software-rendered 3D schematic, not a downloadable mesh or trained network visualization.
- Mouse, touch, keyboard, zoom buttons, component selection, reset/front views, spacing, and label controls operate locally. The renderer schedules frames only when needed and releases listeners, animation frames, and resize observation on navigation. There is no automatic rotation or automatic tour playback. Whole-model/layer switching resets the camera while retaining the selected component.
- A native component menu and textual descriptions remain available when canvas is unavailable. The attention calculator also works without canvas. Whole-model views use E1–E6 and D1–D6 labels, with component names in detail views; the inspector retains complete names and explanations.
- The original model has six encoder layers and six decoder layers. The default scene contains all twelve layers and 63 component blocks: 24 encoder blocks, 36 decoder blocks, two embedding/position blocks, and one vocabulary head. Each numbered layer opens a detail view with explicit input/output context and residual connections. All six cross-attention links originate at the final encoder output. Attention block faces have eight head divisions. A Previous/Next tour visits each real component; context boxes are not selectable model components.
- No runtime dependency, external script, CDN, model download, or remote API is added. No model training code or server access control changes are part of this feature.

HeyGen was requested for a separate eight-second conceptual introduction. That service produces video rather than an interactive mesh. Its player is delivered separately in the development conversation; the application does not embed a pending asset or depend on that service. The interactive schematic itself is code-native. A future bundled video must be reviewed and clearly labeled as conceptual rather than a literal architecture or observed model activation.

## Design reference

The user supplied [Transformers explained in under 3 minutes — BidirectionalAI](https://www.youtube.com/shorts/WmjtFdxTF0Q). The encoder/decoder diagram and its repeated-layer explanation informed the expanded navigation. Geometry, code, styling, and explanations here are original; no frames or video assets were copied. The existing dark navy, amber, mint, blue, and violet scheme is retained.

## Validation — November 15, 2026

| Check | Result |
|---|---|
| `python build_course.py` | 59 lessons and 17 reference chapters |
| `node tests/check_course_dom.cjs` | 76 lesson/reference routes and 69 Mermaid diagrams; existing saved progress retained |
| `node tests/check_transformer.cjs` | 12-layer topology, six final-encoder memory links, 30 residual paths, every layer detail view, 63-component tour, numerical reference cases, perspective, orbit/pan/zoom, touch simulation, keyboard, fallback, and cleanup passed |
| `python -m unittest discover -s tests -p test_server.py -v` | 20 tests passed, including origin/token/host and traversal protections |
| `python -m unittest discover -s tests -p test_content_audit.py -v` | 8 tests passed in the existing test environment |

The Node checks need the existing developer `jsdom` package; they do not add it to the learner's runtime. `SML_JSDOM_PATH` can point to its installed directory. The content audit uses the existing machine-learning test environment.

The canvas tests simulate drawing and input events. The full-stack update builds on commit `78d6d98`, preserving its security/setup changes. Desktop (1,100 px) overview, decoder-layer detail, and narrow (375 px) overview geometry were also rendered by recording the renderer’s canvas operations to SVG and rasterizing them for inspection. All twelve layers, the final-encoder memory connections, layer labels, and detailed residual paths were inspected. This checks drawing output, not browser CSS layout or native input delivery. **Browser layout, visual rendering, and physical touch-device behavior were not verified in this session:** both browser environments could not reach the isolated localhost server. A standalone data-URL preview was rejected by browser URL policy. No workaround was used to bypass that restriction. Full training, model inference, and VM execution were not rerun for this front-end/content change.

To finish a manual visual review, run the lab normally in the same operating system as your browser, open Lesson 59, and exercise orbit, pan, wheel zoom, component selection, both themes, and a narrow window. Confirm the attention output changes from `[1.6044, 1.1978]` to `[1.3395, 0.6605]` when the causal mask is enabled with the initial query. Check that navigating away and back creates only one viewer.
