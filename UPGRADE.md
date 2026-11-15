# Upgrade to Midnight Motion — 58 lessons

Release: `2026.11-midnight.3`.

The lesson sequence now starts at Foundations 1 and continues through lesson 58. Language models starts at 21, Build and retrieve at 35, and Modern language engineering at 43. Saved progress and notes still match their original lessons. Bright blue text is now green, with a darker green in light mode.

## Why your local lab can still show 34 lessons

The original release contained 34 lessons. Later releases added eight Build and retrieve lessons and sixteen Modern language engineering lessons, bringing the total to 58. The count comes from `public/course.json` in the folder served by your running Python process.

A GitHub update or ZIP download does not automatically replace an extracted folder or restart that process. An old launcher can keep port 8765 occupied, so opening the usual browser address still shows the older copy. The server already sends no-cache responses; clearing browser data is not the first fix and can erase progress.

## Windows: switch copies safely

1. Keep the old project folder as a backup. In its launcher window, press **Ctrl+C** to stop the server.
2. Extract the **entire** new ZIP into a separate folder. Do not run inside the ZIP or copy just `index.html`.
3. Open the new folder containing `server.py`, `release.json`, and `start-lab.cmd`.
4. Double-click `start-lab.cmd`. Keep that window open. It now prints the release, lesson count, and exact serving folder.
5. Visit **http://localhost:8765** and refresh. The top bar should show **58 lessons · Midnight Motion**. Click that badge to inspect **About this workspace**.

If you already have a working Python environment, you can reuse its interpreter in place. Do not move or copy the environment itself; virtual environments can contain absolute paths. In PowerShell, substitute the real old folder below:

```powershell
$env:SML_LAB_PYTHON = 'C:\path\to\old-lab\.venv\Scripts\python.exe'
.\start-lab.cmd
```

In Command Prompt:

```bat
set "SML_LAB_PYTHON=C:\path\to\old-lab\.venv\Scripts\python.exe"
start-lab.cmd
```

The old environment and old folder must remain available while you reuse it. Alternatively, use Package setup in the new copy to create a separate environment.

## Preserve your work

Browser lesson progress, evidence notes, and workbench text remain under the same browser profile and exact origin. Keep using `http://localhost:8765`; `127.0.0.1` or a different port is a separate origin. Do not clear site data.

With both launchers stopped, copy `lab/outputs`, `lab/inputs`, and `runs` from the old project to the new project only if their destination directories are absent or empty. Keep the originals. If either destination already has work, keep both copies and resolve individual filename conflicts deliberately; do not bulk-overwrite.

## If the count is still wrong

- **34 lessons and no Midnight badge:** you are still reaching the old server/folder. Stop its launcher and reopen the new one. Inspect the printed serving folder.
- **Port already in use:** stop the older launcher using that port. For a separate temporary workspace, use `start-lab.cmd --port 8766`. That address will have separate browser progress.
- **Midnight badge but a catalog mismatch:** About this workspace shows the actual and expected counts. In the updated folder, run `python build_course.py` using your installed Python, then restart the launcher.
- **Linux/macOS:** follow the same backup and stop/restart procedure, then run `./start-lab.sh`. The README has platform setup details.

## What's new visually

Midnight navy is the default, with an optional light theme; cyan controls, violet examples, mint evidence, and amber cautions share the code-editor palette. The sidebar groups and searches lessons. The homepage offers progress, continuation, and four track cards. Lessons have an outline, highlighted code with copy buttons, and an expandable diagram explorer. The workbench includes highlighted editable code, line numbers, and a resizable console. Results summarize real metrics and compare compatible reports. The research page filters 47 paper and benchmark references by text and topic.

Attention and expert-routing lessons include optional, stepwise teaching animations. They do not autoplay and respect reduced-motion preferences. They are conceptual illustrations, not captured activations. The accepted HeyGen clip is now bundled as an approximately 1.2 MB local MP4 background. It loops silently, is prominent on the homepage, and is dimmed behind lessons. Pause motion / Play motion in the top bar remembers your preference. Reduced-motion settings suppress autoplay unless you explicitly choose Play. The background needs no HeyGen connection.
