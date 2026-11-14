"""SML Lab: small, readable inference, evaluation, retrieval and LoRA exercises.
Run commands from this folder. Downloaded models remain in the HF cache.
"""
import argparse
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LABELS = {"environment", "memory", "data"}
SYSTEM = ('Classify the ML support ticket. environment means installation, imports, '
          'or Python configuration. memory means RAM or GPU memory exhaustion. '
          'data means dataset content, schema, or labels. '
          'Reply with exactly one JSON object: {"label":"environment"}, '
          '{"label":"memory"}, or {"label":"data"}.')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_rows(split):
    return [json.loads(line) for line in (ROOT / "data" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_label(text):
    try:
        value = json.loads(text.strip())
        if isinstance(value, dict) and set(value) == {"label"} and value["label"] in LABELS:
            return value["label"]
    except (ValueError, TypeError):
        pass
    return None


def resolve(model_id):
    path = ROOT / "model-lock.json"
    locks = json.loads(path.read_text()) if path.exists() else {}
    if model_id not in locks:
        raise SystemExit(f'First run: python lab.py lock --model {model_id}')
    return locks[model_id]


def lock(args):
    from huggingface_hub import model_info
    path = ROOT / "model-lock.json"
    locks = json.loads(path.read_text()) if path.exists() else {}
    if args.model in locks and not args.refresh:
        print(f"Already pinned: {args.model}@{locks[args.model]}")
        return
    locks[args.model] = model_info(args.model).sha
    write_json(path, locks)
    print(f"Pinned {args.model}@{locks[args.model]}")


def device_for(name):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Run doctor; use --device cpu for the first labs.")
    return name


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    revision = resolve(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = device_for(args.device)
    # FP32 is deliberately simple and works on CPUs and older GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, torch_dtype=torch.float32,
        trust_remote_code=False, use_safetensors=True).to(device)
    if args.adapter:
        from peft import PeftModel
        meta = json.loads((Path(args.adapter) / "lab_metadata.json").read_text())
        if meta["model"] != args.model or meta["revision"] != revision:
            raise SystemExit("Adapter and pinned base differ. Use the original model-lock.json.")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer, device


def messages(text, system):
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def generate(model, tokenizer, device, text, system, max_new_tokens=64, temperature=0):
    import torch
    inputs = tokenizer.apply_chat_template(messages(text, system), tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    limit = getattr(model.config, "max_position_embeddings", 8192)
    if inputs["input_ids"].shape[-1] + max_new_tokens > limit:
        raise ValueError("Prompt plus output exceeds the model context limit.")
    options = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                   pad_token_id=tokenizer.pad_token_id)
    if temperature > 0:
        options.update(temperature=temperature, top_p=0.9)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new = output[0, inputs["input_ids"].shape[-1]:]
    return {"text": tokenizer.decode(new, skip_special_tokens=True),
            "new_tokens": len(new), "seconds": elapsed,
            "tokens_per_second": len(new) / max(elapsed, 1e-9)}


def doctor(args):
    import torch
    print(json.dumps({"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))


def tokenize(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=resolve(args.model),
                                              trust_remote_code=False)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print("Token IDs:", ids)
    print("Pieces:", tokenizer.convert_ids_to_tokens(ids))
    print("Round trip:", tokenizer.decode(ids))
    print("Chat formatting:")
    print(tokenizer.apply_chat_template(messages(args.prompt, "You are a helpful assistant."),
        tokenize=False, add_generation_prompt=True))


def infer(args):
    model, tokenizer, device = load(args)
    result = generate(model, tokenizer, device, args.prompt,
        "You are a helpful assistant. Give a short answer.", args.max_new_tokens, args.temperature)
    print(json.dumps(result, indent=2))


def evaluate(args):
    model, tokenizer, device = load(args)
    # Warm up so the first request's setup is not mixed into the timing sample.
    generate(model, tokenizer, device, "Hello", "Be brief.", 4)
    results = []
    for row in read_rows(args.split):
        result = generate(model, tokenizer, device, row["text"], SYSTEM, args.max_new_tokens)
        prediction = parse_label(result["text"])
        result.update(id=row["id"], expected=row["label"], prediction=prediction,
                      correct=prediction == row["label"], valid_json=prediction is not None)
        results.append(result)
        print(row["id"], prediction, "expected", row["label"], flush=True)
    n = len(results)
    report = {"model": args.model, "revision": resolve(args.model), "adapter": args.adapter,
        "split": args.split, "seed": args.seed, "device": device, "n": n,
        "accuracy": sum(r["correct"] for r in results) / n,
        "schema_valid_rate": sum(r["valid_json"] for r in results) / n,
        "median_request_seconds": statistics.median(r["seconds"] for r in results),
        "aggregate_tokens_per_second": sum(r["new_tokens"] for r in results) /
                                       sum(r["seconds"] for r in results),
        "predictions": results}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


def encode_example(tokenizer, row, max_length):
    prompt = messages(row["text"], SYSTEM)
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + [{"role": "assistant",
        "content": json.dumps({"label": row["label"]}, separators=(",", ":"))}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError("Template boundary changed: inspect prefix/full token IDs before training.")
    if len(full) > max_length:
        raise ValueError(f'{row["id"]}: {len(full)} tokens exceeds --max-length; avoid silent truncation.')
    labels = [-100] * len(prefix) + full[len(prefix):]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError("No supervised answer tokens.")
    return {"input_ids": full, "labels": labels}


def collate(examples, pad_id, device):
    import torch
    width = max(len(e["input_ids"]) for e in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        n = len(e["input_ids"])
        batch["input_ids"].append(e["input_ids"] + [pad_id] * (width - n))
        batch["attention_mask"].append([1] * n + [0] * (width - n))
        batch["labels"].append(e["labels"] + [-100] * (width - n))
    return {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in batch.items()}


def validation_loss(model, encoded, tokenizer, device):
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for example in encoded:
            batch = collate([example], tokenizer.pad_token_id, device)
            tokens = int((batch["labels"][:, 1:] != -100).sum())
            total += float(model(**batch).loss) * tokens
            count += tokens
    return total / count


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    if args.adapter:
        raise SystemExit("Start each training experiment from the same unmodified base.")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is not empty. Choose a new --output for this experiment.")
    model, tokenizer, device = load(args)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=8,
        lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
    model.config.use_cache = False
    model.print_trainable_parameters()
    train_data = [encode_example(tokenizer, r, args.max_length) for r in read_rows("train")]
    val_data = [encode_example(tokenizer, r, args.max_length) for r in read_rows("validation")]
    before = validation_loss(model, val_data, tokenizer, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    log = []
    model.train()
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        # With-replacement sampling is explicit; --steps means optimizer updates.
        selected = [rng.choice(train_data) for _ in range(args.batch_size)]
        batch = collate(selected, tokenizer.pad_token_id, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Check data and learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        item = {"step": step, "loss": float(loss.detach())}
        log.append(item)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(item), flush=True)
    elapsed = time.perf_counter() - start
    after = validation_loss(model, val_data, tokenizer, device)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    metadata = {"model": args.model, "revision": resolve(args.model), "seed": args.seed,
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "validation_loss_before": before, "validation_loss_after": after,
        "training_seconds": elapsed, "device": device, "loss_log": log,
        "training_note": "Synthetic classroom data; no production quality claim."}
    write_json(out / "lab_metadata.json", metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "loss_log"}, indent=2))


def retrieve(query, docs, k=2):
    # Transparent lexical baseline, intentionally not vector similarity or BM25.
    stop = {"the", "a", "an", "is", "are", "to", "of", "and", "for", "in", "my", "i", "how"}
    words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower())) - stop
    q = words(query)
    ranked = sorted(((len(q & words(d["text"])), d) for d in docs),
                    key=lambda pair: (-pair[0], pair[1]["id"]))
    return [d for score, d in ranked if score > 0][:k]


def rag(args):
    docs = json.loads((ROOT / "data" / "knowledge.json").read_text())
    selected = retrieve(args.prompt, docs)
    print("Retrieved:", [d["id"] for d in selected])
    if not selected:
        print("No lexical matches. I do not have evidence in these notes.")
        return
    if args.retrieve_only:
        print(json.dumps(selected, indent=2))
        return
    model, tokenizer, device = load(args)
    evidence = "\n".join(f'[{d["id"]}] {d["text"]}' for d in selected)
    answer = generate(model, tokenizer, device, f"Notes:\n{evidence}\nQuestion: {args.prompt}",
        "Answer only from the supplied notes. Cite the note ID. If unsupported, say you do not know.",
        args.max_new_tokens)
    print(json.dumps(answer, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "lock", "tokenize", "infer", "eval", "train", "rag"]:
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--adapter")
        if name == "lock":
            p.add_argument("--refresh", action="store_true")
        if name in ["tokenize", "infer", "rag"]:
            p.add_argument("--prompt", default="Explain a neural network in one sentence.")
        if name in ["infer", "eval", "rag"]:
            p.add_argument("--max-new-tokens", type=int, default=48)
        if name == "infer":
            p.add_argument("--temperature", type=float, default=0.0)
        if name == "rag":
            p.add_argument("--retrieve-only", action="store_true")
        if name == "eval":
            p.add_argument("--split", choices=["validation", "test"], default="validation")
            p.add_argument("--output", default="outputs/baseline-validation.json")
        if name == "train":
            p.add_argument("--steps", type=int, default=60)
            p.add_argument("--batch-size", type=int, default=1)
            p.add_argument("--max-length", type=int, default=256)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--output", default="outputs/adapter-60")
    args = parser.parse_args()
    for field in ["threads", "steps", "batch_size", "max_length", "max_new_tokens"]:
        if hasattr(args, field) and getattr(args, field) < 1:
            parser.error(f"{field} must be positive")
    if hasattr(args, "lr") and args.lr <= 0:
        parser.error("lr must be positive")
    if hasattr(args, "temperature") and args.temperature < 0:
        parser.error("temperature cannot be negative")
    functions = dict(doctor=doctor, lock=lock, tokenize=tokenize, infer=infer,
                     eval=evaluate, train=train, rag=rag)
    functions[args.command](args)


if __name__ == "__main__":
    main()
