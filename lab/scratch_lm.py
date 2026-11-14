"""Train a very small causal byte Transformer from random weights on synthetic text.
This teaches pretraining mechanics; it is not a useful general assistant.
"""
import argparse
import json
import math
import random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mechanisms import ToySSM, ToyMoE


class TinyLM(nn.Module):
    def __init__(self, width=64, context=64):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(256, width)
        self.position = nn.Embedding(context, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 4,
                    dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)

    def forward(self, ids):
        length = ids.shape[1]
        if length > self.context:
            raise ValueError("Sequence is longer than the learned position table.")
        x = self.token(ids) + self.position(torch.arange(length, device=ids.device))
        # True above the diagonal means a position cannot see future bytes.
        mask = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), 1)
        return self.head(self.norm(self.blocks(x, mask=mask)))


class ExperimentalLM(nn.Module):
    """Toy SSM, attention plus MoE, or SSM plus attention; not vendor replicas."""
    def __init__(self, kind, width=64, context=64):
        super().__init__()
        self.kind, self.context = kind, context
        self.token = nn.Embedding(256, width)
        self.position = nn.Embedding(context, width)
        self.norm1 = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.mixers = nn.ModuleList([
            ToySSM(width) if kind == "ssm" or (kind == "hybrid" and i == 0)
            else nn.MultiheadAttention(width, 4, batch_first=True) for i in range(2)])
        self.ffns = nn.ModuleList([ToyMoE(width) if kind == "moe" else
            nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4, width))
            for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)

    def forward(self, ids):
        n = ids.shape[1]
        if n > self.context:
            raise ValueError("Sequence is longer than context.")
        x = self.token(ids) + self.position(torch.arange(n, device=ids.device))
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=ids.device), 1)
        for norm1, norm2, mixer, ffn in zip(self.norm1, self.norm2, self.mixers, self.ffns):
            z = norm1(x)
            if isinstance(mixer, ToySSM):
                z, _ = mixer(z)
            else:
                z, _ = mixer(z, z, z, attn_mask=mask, need_weights=False)
            x = x + z
            z = ffn(norm2(x))
            x = x + (z[0] if isinstance(z, tuple) else z)
        return self.head(self.norm(x))


def corpus(seed, n=500):
    rng = random.Random(seed)
    phrases = ["a model predicts the next token", "training changes weights",
               "validation measures generalization", "a tokenizer turns text into numbers",
               "a gradient points toward increasing loss", "an optimizer updates parameters"]
    return "\n".join(f"note {rng.randrange(10000)}: {rng.choice(phrases)}." for _ in range(n))


def batch(data, batch_size, context, generator):
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x, y


def ce(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--architecture", choices=["transformer", "ssm", "moe", "hybrid"], default="transformer")
    parser.add_argument("--output", default="outputs/scratch")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose a new output directory.")
    model = TinyLM() if args.architecture == "transformer" else ExperimentalLM(args.architecture)
    train = torch.tensor(list(corpus(1).encode()), dtype=torch.long)
    val = torch.tensor(list(corpus(2).encode()), dtype=torch.long)
    train_rng = torch.Generator().manual_seed(42)
    val_rng = torch.Generator().manual_seed(99)
    vx, vy = batch(val, 16, 64, val_rng)
    with torch.no_grad():
        before = float(ce(model, vx, vy))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    history = []
    for step in range(1, args.steps + 1):
        x, y = batch(train, 8, 64, train_rng)
        loss = ce(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            history.append({"step":step, "loss":float(loss.detach())})
            print(history[-1], flush=True)
    model.eval()
    with torch.no_grad():
        after = float(ce(model, vx, vy))
        ids = torch.tensor([list(b"note 42:")], dtype=torch.long)
        for _ in range(120):
            logits = model(ids[:, -64:])[:, -1, :]
            nxt = torch.multinomial(torch.softmax(logits / 0.8, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
    generated = bytes(ids[0].tolist()).decode("utf-8", errors="replace")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "weights.pt")
    report = {"architecture":args.architecture, "parameters":sum(p.numel() for p in model.parameters()),
        "validation_loss_before":before, "validation_loss_after":after,
        "byte_perplexity_after":math.exp(after), "sample":generated, "history":history,
        "scope":"Same synthetic grammar, distinct generated corpus; not real-language generalization."}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
