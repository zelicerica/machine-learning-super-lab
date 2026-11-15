"""Small numerical experiments, not pretrained or production model implementations."""
import argparse
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


def quantize_demo(qmax=127):
    if qmax not in (7,127):
        raise ValueError('Use qmax 7 or 127; both are stored in int8 in this demonstration.')
    weights = torch.randn(64, 64)
    scale = weights.abs().max().clamp_min(1e-8) / qmax
    integers = torch.round(weights / scale).clamp(-qmax, qmax).to(torch.int8)
    restored = integers.float() * scale
    return {"integer_range": [-qmax,qmax], "scale":float(scale),
            "storage_dtype":"int8; no four-bit packing",
            "fp32_weight_bytes": weights.numel() * weights.element_size(),
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


def math_demo():
    weight=torch.tensor(1.,requires_grad=True)
    loss=(weight*2-6).square();loss.backward()
    updated=weight.detach()-.1*weight.grad
    return {"token_ids_shape":[2,16],"embedding_shape":[2,16,576],
        "logits":[2.,1.,0.],"softmax":torch.tensor([2.,1.,0.]).softmax(-1).tolist(),
        "negative_log_likelihood":{"p_0.1":float(-torch.tensor(.1).log()),"p_0.8":float(-torch.tensor(.8).log())},
        "one_weight_update":{"initial_weight":1.,"input":2.,"target":6.,"initial_squared_error":float(loss.detach()),
            "gradient":float(weight.grad),"learning_rate":.1,"updated_weight":float(updated),"new_squared_error":float((updated*2-6).square())}}


def architecture_demo():
    x = torch.randn(2, 12, 16)
    moe = ToyMoE()
    y, counts = moe(x)
    ssm = ToySSM()
    z, state = ssm(x)
    return {"moe_total_parameters":sum(p.numel() for p in moe.parameters()),
        "moe_tokens_per_expert":counts.tolist(), "moe_output_shape":list(y.shape),
        "ssm_output_shape":list(z.shape), "ssm_final_state_shape":list(state.shape)}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',choices=['all','math','quantization','distillation','architectures'],default='all')
    parser.add_argument('--qmax',type=int,choices=[7,127],default=127)
    args=parser.parse_args()
    if args.qmax!=127 and args.demo!='quantization':parser.error('--qmax applies to --demo quantization only.')
    torch.manual_seed(42)
    demonstrations={'math':math_demo,'quantization':lambda:quantize_demo(args.qmax),
        'distillation':distillation_demo,'architectures':architecture_demo}
    selected=demonstrations if args.demo=='all' else {args.demo:demonstrations[args.demo]}
    print(json.dumps({name:fn() for name,fn in selected.items()},indent=2))
