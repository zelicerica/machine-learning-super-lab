"""Inspectable small Transformer components. Educational CPU implementations.

No pretrained weights, external kernels, or network downloads are needed here.
Latent attention and attention residuals are simplified mechanism experiments,
not faithful reproductions of DeepSeek or Kimi models.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    width: int = 32
    heads: int = 4
    kv_heads: int = 4
    layers: int = 2
    context: int = 96
    position: str = "learned"
    norm: str = "layer"
    qk_norm: bool = False
    swiglu: bool = False
    window: int = 0
    latent: bool = False
    experts: int = 0
    depth_attention: bool = False

    def __post_init__(self):
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("Width must divide into heads; key/value heads must divide query heads.")
        if (self.width // self.heads) % 2:
            raise ValueError("Use an even head dimension for rotary positions.")
        if self.position not in {"learned", "rope", "alibi"} or self.norm not in {"layer", "rms"}:
            raise ValueError("Unknown position or normalization choice.")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__(); self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.scale


def rotary(x, positions):
    dimension = x.shape[-1]
    frequency = 10000.0 ** (-torch.arange(0, dimension, 2, device=x.device).float() / dimension)
    angles = positions.float()[:, None] * frequency[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c; self.dimension = c.width // c.heads
        self.query = nn.Linear(c.width, c.width, bias=False)
        if c.latent:
            self.latent_width = c.width // 4
            self.compress = nn.Linear(c.width, self.latent_width, bias=False)
            self.expand = nn.Linear(self.latent_width, 2 * c.kv_heads * self.dimension, bias=False)
        else:
            self.key_value = nn.Linear(c.width, 2 * c.kv_heads * self.dimension, bias=False)
        self.output = nn.Linear(c.width, c.width, bias=False)
        if c.qk_norm:
            self.query_norm = RMSNorm(self.dimension); self.key_norm = RMSNorm(self.dimension)
        # Four-head ALiBi slopes for this teaching model; powers of two only.
        self.register_buffer("slopes", 2.0 ** (-8.0 * torch.arange(1, c.heads + 1) / c.heads))
        self.last_weights = None

    def forward(self, x, cache=None, capture=False):
        batch, length, _ = x.shape; c = self.c
        offset = 0 if cache is None else cache["total"]
        query_positions = torch.arange(offset, offset + length, device=x.device)
        query = self.query(x).view(batch, length, c.heads, self.dimension).transpose(1, 2)
        if c.latent:
            latent = self.compress(x)
            if cache is not None: latent = torch.cat((cache["latent"], latent), 1)
            kv = self.expand(latent).view(batch, -1, 2, c.kv_heads, self.dimension)
            key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
            key_positions = torch.arange(offset + length - key.shape[2], offset + length, device=x.device)
            if c.position == "rope": key = rotary(key, key_positions)
            updated = {"latent": latent, "total": offset + length}
        else:
            kv = self.key_value(x).view(batch, length, 2, c.kv_heads, self.dimension)
            key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
            if c.position == "rope": key = rotary(key, query_positions)
            if cache is not None:
                key = torch.cat((cache["key"], key), 2); value = torch.cat((cache["value"], value), 2)
            key_positions = torch.arange(offset + length - key.shape[2], offset + length, device=x.device)
            updated = {"key": key, "value": value, "total": offset + length}
        if c.position == "rope": query = rotary(query, query_positions)
        if c.qk_norm: query, key = self.query_norm(query), self.key_norm(key)
        repetitions = c.heads // c.kv_heads
        key = key.repeat_interleave(repetitions, 1); value = value.repeat_interleave(repetitions, 1)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.dimension)
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = distance >= 0
        if c.window: allowed = allowed & (distance < c.window)
        if c.position == "alibi": scores = scores - self.slopes[None, :, None, None] * distance.clamp_min(0)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        if capture: self.last_weights = weights.detach().cpu()
        result = (weights @ value).transpose(1, 2).reshape(batch, length, c.width)
        if c.window:
            for name in ("key", "value"):
                if name in updated: updated[name] = updated[name][:, :, -c.window:].contiguous()
            if "latent" in updated: updated["latent"] = updated["latent"][:, -c.window:].contiguous()
        return self.output(result), updated


class FeedForward(nn.Module):
    def __init__(self, width, gated=False):
        super().__init__(); self.gated = gated
        # Similar parameter budgets, not exactly equal after rounding.
        hidden = (8 * width // 3) if gated else 4 * width
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        if gated: self.gate = nn.Linear(width, hidden, bias=False)

    def forward(self, x):
        h = F.silu(self.gate(x)) * self.up(x) if self.gated else F.gelu(self.up(x))
        return self.down(h)


class Experts(nn.Module):
    def __init__(self, width, count):
        super().__init__(); self.router = nn.Linear(width, count)
        self.experts = nn.ModuleList([FeedForward(width) for _ in range(count)])
        self.counts = None; self.balance_loss = None

    def forward(self, x):
        probs = self.router(x).softmax(-1); chosen = probs.argmax(-1)
        result = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            selected = chosen == i
            if selected.any(): result[selected] = expert(x[selected]) * probs[selected][:, i:i+1]
        fractions = torch.stack([(chosen == i).float().mean() for i in range(len(self.experts))])
        self.counts = torch.bincount(chosen.flatten(), minlength=len(self.experts)).detach()
        self.balance_loss = len(self.experts) * (fractions * probs.mean((0, 1))).sum()
        return result


class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); normalize = nn.LayerNorm if c.norm == "layer" else RMSNorm
        self.norm1 = normalize(c.width); self.norm2 = normalize(c.width)
        self.attention = Attention(c)
        self.feed_forward = Experts(c.width, c.experts) if c.experts else FeedForward(c.width, c.swiglu)

    def forward(self, x, cache=None, capture=False):
        update, cache = self.attention(self.norm1(x), cache, capture)
        x = x + update
        return x + self.feed_forward(self.norm2(x)), cache


class TinyTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__(); self.config = config or Config(); c = self.config
        self.embedding = nn.Embedding(256, c.width)
        if c.position == "learned": self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width) if c.norm == "layer" else RMSNorm(c.width)
        self.head = nn.Linear(c.width, 256, bias=False)
        if c.depth_attention:
            self.depth_queries = nn.Parameter(torch.randn(c.layers, c.width) * .02)

    def forward(self, ids, caches=None, capture=False):
        offset = 0 if caches is None else caches[0]["total"]
        if offset + ids.shape[1] > self.config.context: raise ValueError("Context limit exceeded; shorten the input or generation.")
        x = self.embedding(ids)
        if self.config.position == "learned":
            x = x + self.position(torch.arange(offset, offset + ids.shape[1], device=ids.device))
        updated = []; history = [x]
        for i, block in enumerate(self.blocks):
            if self.config.depth_attention:
                # A block-level depth-mixture experiment, not paper-exact AttnRes.
                values = torch.stack(history, 2)
                scores = (F.normalize(values, dim=-1) * self.depth_queries[i]).sum(-1)
                x = (scores.softmax(-1)[..., None] * values).sum(2)
            x, cache = block(x, None if caches is None else caches[i], capture)
            history.append(x); updated.append(cache)
        return self.head(self.norm(x)), updated


class AdapterLinear(nn.Module):
    """LoRA with either a float base or genuinely packed uniform four-bit weights.

    Uniform quantization is NOT the NormalFloat4 algorithm from QLoRA. We unpack
    for computation on CPU; this teaches storage/adaptation, not fast low-bit math.
    """
    def __init__(self, original, rank=4, quantized=False):
        super().__init__(); self.in_features = original.in_features; self.out_features = original.out_features
        self.rank = rank; self.quantized = quantized
        weight = original.weight.detach().clone()
        if original.bias is not None: raise ValueError("This adapter targets bias-free projections only.")
        if quantized:
            scale = weight.abs().amax(1, keepdim=True).clamp_min(1e-8) / 7
            codes = (weight / scale).round().clamp(-7, 7).to(torch.int16) + 8
            if codes.numel() % 2: raise ValueError("Use an even weight element count.")
            flat = codes.flatten().to(torch.uint8)
            self.register_buffer("packed", flat[0::2] | (flat[1::2] << 4)); self.register_buffer("scale", scale)
        else: self.register_buffer("base_weight", weight)
        self.a = nn.Parameter(torch.randn(rank, self.in_features) * .02)
        self.b = nn.Parameter(torch.zeros(self.out_features, rank))

    def weight(self):
        if not self.quantized: return self.base_weight
        codes = torch.stack((self.packed & 15, self.packed >> 4), 1).flatten()
        return (codes.to(self.scale.dtype).view(self.out_features, self.in_features) - 8) * self.scale

    def forward(self, x):
        return F.linear(x, self.weight()) + F.linear(F.linear(x, self.a), self.b)


def attach_adapters(model, quantized=False):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for block in model.blocks:
        block.attention.query = AdapterLinear(block.attention.query, quantized=quantized)
        block.attention.output = AdapterLinear(block.attention.output, quantized=quantized)
    model.head = AdapterLinear(model.head, quantized=quantized)
    return model


def cache_bytes(caches):
    return sum(t.numel() * t.element_size() for c in caches for t in c.values() if torch.is_tensor(t))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def state_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def model_payload(model, adapter=None):
    return {"config": asdict(model.config), "state": model.state_dict(), "adapter": adapter}


def restore(payload):
    model = TinyTransformer(Config(**payload["config"]))
    if payload.get("adapter"): attach_adapters(model, payload["adapter"] == "packed4")
    model.load_state_dict(payload["state"]); model.eval()
    return model
