"""Write original, synthetic classroom tickets; no external dataset required."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"
# Phrasings are deliberately assigned to splits before training.
# Related concepts still occur across splits: this is a tiny teaching benchmark.
DATA = {
"train": {
"environment": [
"Python cannot import torch after installation.", "pip installed into a different interpreter.",
"The virtual environment is not activated.", "The transformers package cannot be found.",
"My notebook uses the wrong Python kernel.", "The installed package versions conflict.",
"Python is not recognized in the terminal.", "The interpreter path points outside the project.",
"Importing peft raises ModuleNotFoundError.", "I need to select the project Python executable.",
"Installation failed because the Python version is unsupported.", "The package dependency resolver reports incompatible versions."],
"memory": [
"CUDA out of memory while training.", "The GPU cannot fit the batch in VRAM.",
"Loading model weights exhausts system RAM.", "Increasing batch size crashes due to memory exhaustion.",
"A long context fills the GPU memory.", "Training activations use all available VRAM.",
"The operating system kills Python after RAM fills up.", "My model allocation needs more memory than available.",
"The optimizer states do not fit in GPU memory.", "Generation exhausts the KV cache memory budget.",
"A larger model makes the process run out of RAM.", "There is insufficient memory to allocate another tensor."],
"data": [
"The dataset has missing labels.", "The JSONL file contains malformed rows.",
"Train and validation contain duplicate examples.", "The CSV column names do not match the schema.",
"Some examples have empty response fields.", "The labels contain spelling inconsistencies.",
"The evaluation answers leaked into the training set.", "The dataset contains incorrectly tagged tickets.",
"A training row is not valid JSON.", "The input text field contains null values.",
"The data split accidentally includes the same document twice.", "The answer column is missing from the dataset."]},
"validation": {
"environment": ["The shell and editor are running different Python installations.",
"An import fails even though pip says the library is installed.",
"The notebook kernel cannot see the packages in my venv.",
"Two required libraries demand incompatible dependency versions."],
"memory": ["The accelerator has no free space left for tensors.",
"Reducing sequence length stops my VRAM allocation failure.",
"The process is terminated when its resident memory reaches the machine limit.",
"The weights load, but training needs more GPU memory than is free."],
"data": ["Several annotations disagree with the actual ticket content.",
"My held out split repeats records from the fitting split.",
"The parser rejects a record with an unterminated JSON string.",
"The examples have a target field, but the loader expects label."]},
"test": {
"environment": ["The interpreter selected by the IDE is not the one where dependencies were installed.",
"The wheel refuses to install on this Python release.",
"After opening a new terminal, the project libraries are no longer importable.",
"Dependency resolution fails before any model is downloaded."],
"memory": ["A tensor allocation requests two gigabytes but the GPU has only one available.",
"The machine starts swapping heavily as model parameters load.",
"Keeping several long conversations active exhausts the device cache capacity.",
"The forward pass works for one sample but runs out of VRAM for eight."],
"data": ["The same customer document appears in both learning and scoring examples.",
"Some training outputs use category names outside the allowed set.",
"Half the records have blank questions despite passing file parsing.",
"The supposed correct answers were attached to the wrong input rows."]}}

KNOWLEDGE = [
 {"id":"env-1", "text":"When a Python import fails, compare python -m pip --version with the active Python executable. Install into the project virtual environment."},
 {"id":"mem-1", "text":"For CUDA out of memory, first reduce batch size to one, then shorten sequence length. LoRA reduces trainable parameter storage, but activations still use memory."},
 {"id":"data-1", "text":"For missing labels or malformed JSON, validate each dataset record and remove or repair invalid rows before training."},
 {"id":"eval-1", "text":"Keep test examples separate from training and hyperparameter selection. Use validation to choose settings."}]

if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    seen = set()
    for split, labels in DATA.items():
        rows = []
        for label, examples in labels.items():
            for i, text in enumerate(examples):
                assert text.lower() not in seen
                seen.add(text.lower())
                rows.append({"id":f"{split}-{label}-{i:02d}", "text":text, "label":label})
        (ROOT / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(split, len(rows))
    (ROOT / "knowledge.json").write_text(json.dumps(KNOWLEDGE, indent=2))
