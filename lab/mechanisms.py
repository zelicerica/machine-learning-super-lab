"""Small numerical experiments, not pretrained or production model implementations."""
import json
import torch
from torch import nn
from torch.nn import functional as F


class ToyMoE(nn.Module):
    """Top-1 routing; only routed experts execute. No production capacity handling."""
    def __init__(self, width=16, experts=4):
        super().__init__()
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, 32), nn.GELU(),
            nn.Linear(32, width)) for _ in range(experts)])

    def forward(self, x):
        probabilities = self.router(x).softmax(-1)
        chosen = probabilities.argmax(-1)
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            selected = chosen == i
            if selected.any():
                y[selected] = expert(x[selected]) * probabilities[selected][:, i:i+1]
        return y, torch.bincount(chosen.flatten(), minlength=len(self.experts))


class ToySSM(nn.Module):
    """Diagonal linear recurrence; intentionally NOT Mamba's selective architecture."""
    def __init__(self, width=16):
        super().__init__()
        self.decay_logits = nn.Parameter(torch.zeros(width))
        self.input = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x, state=None):
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[2])
        ys = []
        decay = torch.sigmoid(self.decay_logits)
        for token in x.unbind(1):
            state = decay * state + self.input(token)
            ys.append(self.output(state))
        return torch.stack(ys, dim=1), state


def quantize_demo():
    weights = torch.randn(64, 64)
    scale = weights.abs().max().clamp_min(1e-8) / 127
    integers = torch.round(weights / scale).clamp(-127, 127).to(torch.int8)
    restored = integers.float() * scale
    return {"fp32_weight_bytes": weights.numel() * weights.element_size(),
            "int8_weight_bytes_plus_scale": integers.numel() + 4,
            "mean_squared_error": float((weights - restored).square().mean())}


def distillation_demo():
    teacher_logits = torch.tensor([[4., 2., 0.]])
    student_logits = torch.tensor([[1., 1., 1.]], requires_grad=True)
    temperature = 2.0
    teacher_probabilities = (teacher_logits / temperature).softmax(-1)
    loss = F.kl_div((student_logits / temperature).log_softmax(-1),
                    teacher_probabilities, reduction="batchmean") * temperature**2
    loss.backward()
    return {"teacher_probabilities": teacher_probabilities.tolist(),
            "student_gradient": student_logits.grad.tolist(), "loss":float(loss.detach())}


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 12, 16)
    moe = ToyMoE()
    y, counts = moe(x)
    ssm = ToySSM()
    z, state = ssm(x)
    print(json.dumps({"quantization":quantize_demo(), "distillation":distillation_demo(),
        "moe_total_parameters":sum(p.numel() for p in moe.parameters()),
        "moe_tokens_per_expert":counts.tolist(), "moe_output_shape":list(y.shape),
        "ssm_output_shape":list(z.shape), "ssm_final_state_shape":list(state.shape)}, indent=2))
