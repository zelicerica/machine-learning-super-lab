"""Modern language-model teaching builds; small, local, and inspectable.

All scores concern original synthetic teaching data. No benchmark leaderboard
claims. See course/Modern-Language-Engineering.md for scope and derivations.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modern_models import Config, TinyTransformer, Experts, attach_adapters, cache_bytes, parameter_count, state_bytes, model_payload, restore

EXPERIMENTS = ("attention", "components", "experts", "research", "tuning", "preference", "reward", "reinforce", "ppo", "grpo", "gspo", "evaluation", "judging", "serving")


def tokens(text): return list(text.encode("utf-8"))


def split_rows(rows, seed=73):
    ordered = list(rows); random.Random(seed).shuffle(ordered)
    cut, end = int(.6 * len(ordered)), int(.8 * len(ordered))
    return {"train": ordered[:cut], "validation": ordered[cut:end], "test": ordered[end:]}


def notes():
    # Split whole sentences, never windows from the same sentence across splits.
    rows = [(f"service {service} event {event}: ", f"{state}.\n")
            for service in ("api", "queue", "search", "web")
            for event, state in (("start", "running"), ("stop", "offline"), ("retry", "pending"), ("restore", "running"))
            for _ in range(1)]
    return split_rows(rows)


def tickets():
    phrases = {"environment": ["cannot import module", "missing package", "wrong python", "package not installed", "module absent", "interpreter mismatch"],
               "memory": ["out of memory", "allocation failed", "batch too large", "memory exhausted", "tensor too large", "device memory full"],
               "data": ["missing label", "invalid column", "empty training file", "label field absent", "wrong data schema", "no training rows"]}
    # Template-level split; identifiers never disguise copies of a template.
    result = {"train": [], "validation": [], "test": []}
    for label, variants in phrases.items():
        for i, phrase in enumerate(variants):
            split = "train" if i < 4 else "validation" if i == 4 else "test"
            result[split].append((f"ticket: {phrase}\nlabel:", " " + label + "\n"))
    return result


def arithmetic():
    return split_rows([(f"{a}+{b}=", f"{a+b:02d}") for a in range(10) for b in range(10)])


def batch(rows, response_only=True):
    encoded = [tokens(p + r) for p, r in rows]; length = max(map(len, encoded)) - 1
    x = torch.zeros(len(rows), length, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, ((prompt, response), ids) in enumerate(zip(rows, encoded)):
        x[i, :len(ids)-1] = torch.tensor(ids[:-1]); y[i, :len(ids)-1] = torch.tensor(ids[1:])
        if response_only: y[i, :len(tokens(prompt))-1] = -100
    return x, y


def sequence_logps(model, rows):
    x, labels = batch(rows)
    logps = model(x)[0].log_softmax(-1)
    mask = labels != -100
    chosen = logps.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (chosen * mask).sum(-1), mask.sum(-1)


def loss_on(model, rows, response_only=True):
    model.eval()
    with torch.no_grad():
        x, y = batch(rows, response_only)
        return float(F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()))


def fit(model, rows, steps, seed=42, lr=.004, response_only=True, balance=.01):
    model.train(); rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []; started = time.perf_counter()
    for step in range(steps):
        selected = [rng.choice(rows) for _ in range(8)]; x, y = batch(selected, response_only)
        logits, _ = model(x); loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        extras = [m.balance_loss for m in model.modules() if isinstance(m, Experts)]
        if extras: loss = loss + balance * torch.stack(extras).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            item = {"step": step + 1, "loss": float(loss.detach())}; history.append(item); print(item, flush=True)
    model.eval(); return history, time.perf_counter() - started


@torch.no_grad()
def generate(model, prompt, count=20, cached=True, sample=False):
    ids = torch.tensor([tokens(prompt)]); caches = None; output = []; times = []
    if not ids.numel() or ids.shape[1] + count > model.config.context: raise ValueError("Prompt must be nonempty and fit the context with output.")
    start = time.perf_counter()
    for i in range(count):
        logits, updated = model(ids[:, -1:] if cached and caches is not None else ids, caches if cached else None)
        p = logits[:, -1].softmax(-1)
        nxt = torch.multinomial(p, 1) if sample else p.argmax(-1, keepdim=True)
        ids = torch.cat((ids, nxt), 1); caches = updated; output.append(int(nxt[0, 0])); times.append(time.perf_counter()-start)
    return {"text": bytes(output).decode("utf-8", errors="replace"), "token_ids": output, "arrival_seconds": times, "cache_bytes": cache_bytes(caches)}


def export_model(out, model, sample, adapter=None, task="completion"):
    model.eval(); file = out / "model.pt"; torch.save(model_payload(model, adapter), file)
    restored = restore(torch.load(file, map_location="cpu", weights_only=True))
    probe = torch.tensor([tokens(sample)])
    with torch.no_grad(): error = float((model(probe)[0] - restored(probe)[0]).abs().max())
    if error > 1e-6: raise RuntimeError("Reload changed model outputs.")
    manifest = {"format": "superlab-modern", "task": task, "config": asdict(model.config), "adapter": adapter,
                "input_contract": "A nonempty UTF-8 text prefix, at most 64 bytes. The arithmetic task expects a+b= with a and b from 0 through 9.",
                "trusted_artifacts_only": True}
    (out / "model-manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "example-input.json").write_text(json.dumps(sample))
    return error


def predict(folder, inputs):
    folder = Path(folder); manifest = json.loads((folder / "model-manifest.json").read_text())
    if not isinstance(inputs, str) or not 0 < len(tokens(inputs)) <= 64: raise ValueError("Enter a text prefix of 1–64 UTF-8 bytes.")
    if manifest["task"] == "reward":
        model = RewardScorer(); model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True)); model.eval()
        with torch.no_grad(): return {"reward_score": float(model([inputs])[0]), "meaning": "Learned relative score, not probability of truth."}
    model = restore(torch.load(folder / "model.pt", map_location="cpu", weights_only=True)); model.eval()
    if manifest["task"] == "arithmetic":
        import re
        if not re.fullmatch(r"[0-9]\+[0-9]=", inputs): raise ValueError("Use the arithmetic format, such as 3+4=.")
        return {"completion": digit_rollout(model, [inputs], greedy=True)[0], "constrained_to": "two decimal digits"}
    return generate(model, inputs, count=min(20, model.config.context-len(tokens(inputs))))


def write_plot(out, rows, metric, title, name="comparison.png"):
    plt.figure(figsize=(8, 4)); plt.bar(list(rows), [r[metric] for r in rows.values()], color="#557a53")
    plt.ylabel(metric.replace("_", " ")); plt.title(title); plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(out/name, dpi=130); plt.close(); return name


def model_experiment(a):
    families = {
        "attention": {"multi-head": Config(), "grouped-query": Config(kv_heads=2), "multi-query": Config(kv_heads=1), "window-8": Config(window=8)},
        "components": {"baseline": Config(), "rotary": Config(position="rope"), "linear-bias": Config(position="alibi"), "root-mean-square": Config(norm="rms"), "query-key-norm": Config(qk_norm=True), "swish-gate": Config(swiglu=True)},
        "experts": {"dense": Config(), "four-experts": Config(experts=4)},
        "research": {"baseline": Config(), "latent-cache": Config(latent=True), "depth-mixture": Config(depth_attention=True)}
    }
    rows = notes(); measurements = {}; all_history = {}; selected = None
    for name, config in families[a.experiment].items():
        torch.manual_seed(a.seed); model = TinyTransformer(config)
        history, seconds = fit(model, rows["train"], a.steps, a.seed, response_only=False)
        validation = loss_on(model, rows[a.split], False)
        probe = "service api event start: "
        cached = generate(model, probe, 16, True); uncached = generate(model, probe, 16, False)
        ids = torch.tensor([tokens(probe)])
        with torch.no_grad(): model(ids, capture=True)
        error = cache_equivalence(model, ids)
        # Cache checks end on one token; measure routing on the full prompt.
        with torch.no_grad(): model(ids, capture=True)
        measurements[name] = {"loss": validation, "byte_perplexity": math.exp(validation), "parameters": parameter_count(model),
                              "training_seconds": seconds, "cache_bytes": cached["cache_bytes"], "cached_max_logit_error": error,
                              "generated": cached["text"], "cached_tokens_equal": cached["token_ids"] == uncached["token_ids"], "config": asdict(config)}
        if config.experts:
            measurements[name]["routing_counts"] = [m.counts.tolist() for m in model.modules() if isinstance(m, Experts)]
            measurements[name]["routing_probe_tokens"] = ids.numel()
            measurements[name]["expert_note"] = "Counts cover the full example prompt, separately per layer. All four experts are stored. One expert per token executes; router and other layers still execute. No distributed dispatch or capacity limits."
        torch.save(model_payload(model), a.out/(name+".pt")); all_history[name] = history; selected = model
    probe = "service api event start: "
    error = export_model(a.out, selected, probe)
    weights = selected.blocks[0].attention.last_weights[0, 0].numpy()
    plt.figure(figsize=(6, 5)); plt.imshow(weights, origin="upper", cmap="viridis"); plt.colorbar(label="attention weight")
    plt.xlabel("Key position"); plt.ylabel("Query position"); plt.title("First layer, first head; example prompt")
    plt.tight_layout(); plt.savefig(a.out/"attention.png", dpi=120); plt.close()
    return {"metrics": measurements, "history": all_history[list(all_history)[-1]], "reload_verified": error <= 1e-6,
            "artifacts": [write_plot(a.out, measurements, "loss", "Held-out byte loss; compare one change at a time"), "attention.png"],
            "limitations": "Sixteen synthetic event sentences; held-out combinations share vocabulary and rules. Latent and depth-mixture variants are mechanism demonstrations, not paper reproductions. Python loops do not predict optimized hardware speed."}


@torch.no_grad()
def cache_equivalence(model, ids):
    model.eval(); full = model(ids)[0]; cache = None; pieces = []
    for i in range(ids.shape[1]):
        logits, cache = model(ids[:, i:i+1], cache); pieces.append(logits)
    return float((full - torch.cat(pieces, 1)).abs().max())


def tuning(a):
    torch.manual_seed(a.seed); base = TinyTransformer(); fit(base, notes()["train"], a.steps, a.seed, response_only=False)
    data = tickets(); measurements = {}; selected = None; history = []
    for mode in ("full", "lora", "packed4"):
        torch.manual_seed(a.seed+1); model = copy.deepcopy(base)
        if mode != "full": attach_adapters(model, mode == "packed4")
        frozen_before = {k: v.clone() for k, v in model.named_buffers()}
        before = loss_on(model, data[a.split]); history, seconds = fit(model, data["train"], a.steps, a.seed, lr=.008)
        after = loss_on(model, data[a.split]); predictions = []
        for prompt, expected in data[a.split]:
            text = generate(model, prompt, 16)["text"]
            predictions.append({"prompt": prompt, "expected": expected.strip(), "generated": text, "correct": text.strip().split("\n")[0] == expected.strip()})
        unchanged = all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in frozen_before.items())
        measurements[mode] = {"loss_before": before, "loss": after, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                              "state_bytes": state_bytes(model), "training_seconds": seconds, "frozen_buffers_unchanged": unchanged,
                              "accuracy": sum(p["correct"] for p in predictions)/len(predictions), "predictions": predictions}
        torch.save(model_payload(model, None if mode == "full" else mode), a.out/(mode+".pt")); selected = model
    export_model(a.out, selected, data[a.split][0][0], "packed4")
    return {"metrics": measurements, "history": history, "reload_verified": True, "artifacts": [write_plot(a.out, measurements, "loss", "Same base and downstream examples")],
            "limitations": "Uniform four-bit packing of selected adapted projections only, with float unpacking during computation. This is not NormalFloat4 QLoRA, full-model quantization, or a memory/speed claim for a GPU kernel. Three held-out examples per split provide very weak accuracy evidence."}


def preference_rows(split):
    return [(p, good, " unknown\n") for p, good in tickets()[split]]


def dpo_loss(policy_good, policy_bad, reference_good, reference_bad, beta=.1):
    return -F.logsigmoid(beta * ((policy_good - policy_bad) - (reference_good - reference_bad))).mean()


def preference(a):
    torch.manual_seed(a.seed); model = TinyTransformer(); data = tickets()
    fit(model, data["train"], a.steps, a.seed); reference = copy.deepcopy(model).eval()
    for p in reference.parameters(): p.requires_grad_(False)
    digest_before = weight_digest(reference)
    def margins(rows):
        with torch.no_grad():
            good, _ = sequence_logps(model, [(p,g) for p,g,b in rows]); bad, _ = sequence_logps(model, [(p,b) for p,g,b in rows])
        return (good-bad).tolist()
    before = margins(preference_rows(a.split)); optimizer = torch.optim.AdamW(model.parameters(), lr=.001); history=[]
    rows = preference_rows("train")
    for step in range(a.steps):
        good = [(p,g) for p,g,b in rows]; bad = [(p,b) for p,g,b in rows]
        pg,_=sequence_logps(model,good); pb,_=sequence_logps(model,bad)
        with torch.no_grad(): rg,_=sequence_logps(reference,good); rb,_=sequence_logps(reference,bad)
        loss=dpo_loss(pg,pb,rg,rb,a.beta); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        if step % 25 == 0 or step+1 == a.steps: history.append({"step":step+1,"loss":float(loss.detach())})
    after=margins(preference_rows(a.split)); export_model(a.out,model,data[a.split][0][0])
    return {"metrics":{"heldout_preference_win_rate_before":float(np.mean(np.array(before)>0)),"heldout_preference_win_rate_after":float(np.mean(np.array(after)>0)),
                       "margins_before":before,"margins_after":after,"reference_weights_unchanged":digest_before==weight_digest(reference),"heldout_response_loss":loss_on(model,data[a.split])},
            "history":history,"reload_verified":True,"limitations":"Synthetic preferred-label versus unknown-response pairs. A higher preference margin does not establish helpfulness, factuality, or free-running generation quality."}


def weight_digest(model):
    h=hashlib.sha256()
    for t in model.state_dict().values():h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RewardScorer(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(256,16); self.score=nn.Sequential(nn.Linear(16,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,texts):
        values=[self.embedding(torch.tensor(tokens(s))).mean(0) for s in texts]
        return self.score(torch.stack(values)).squeeze(-1)


def reward_pairs():
    return {
        'train':[("Check the package version.","I refuse."),("Inspect the input schema.","Not my problem."),("Reduce the batch size.","Whatever."),("Check the Python environment.","Go away.")],
        'validation':[("Review the error message.","I will not help."),("Try a smaller input.","Figure it out yourself.")],
        'test':[("Read the missing field name.","Do not bother me."),("Verify the selected interpreter.","This is your problem.")]}


def reward(a):
    torch.manual_seed(a.seed); model=RewardScorer(); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    pairs=reward_pairs();train=pairs['train'];held=pairs[a.split]
    history=[]
    for step in range(a.steps):
        loss=-F.logsigmoid(model([p for p,n in train])-model([n for p,n in train])).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%25==0 or step+1==a.steps:history.append({"step":step+1,"loss":float(loss.detach())})
    model.eval()
    with torch.no_grad():
        margins=(model([p for p,n in held])-model([n for p,n in held])).tolist()
        scores=model(["Check the package version.","Check the package version. "*8,"The answer is definitely 999."]).tolist()
    torch.save(model.state_dict(),a.out/"model.pt"); reloaded=RewardScorer();reloaded.load_state_dict(torch.load(a.out/"model.pt",weights_only=True));reloaded.eval()
    with torch.no_grad():equal=torch.equal(model([held[0][0]]),reloaded([held[0][0]]))
    (a.out/"model-manifest.json").write_text(json.dumps({"format":"superlab-modern","task":"reward","trusted_artifacts_only":True}))
    (a.out/"example-input.json").write_text(json.dumps(held[0][0]))
    return {"metrics":{"heldout_pair_accuracy":float(np.mean(np.array(margins)>0)),"pair_margins":margins,"original_repeated_and_unchecked_claim_scores":scores},
            "history":history,"reload_verified":equal,"limitations":"Four original teaching preference pairs; style-only scorer with mean pooling. No verification of factual content. Repetition can preserve the score. These are relative scores, not calibrated correctness probabilities."}


def digit_rollout(model, prompts, greedy=False):
    ids=torch.tensor([tokens(p) for p in prompts]); answers=["" for _ in prompts]
    with torch.no_grad():
        for _ in range(2):
            logits=model(ids)[0][:,-1,48:58]; digit=logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            for i,d in enumerate(digit.tolist()):answers[i]+=str(d)
            ids=torch.cat((ids,(digit+48)[:,None]),1)
    return answers


def digit_logps(model,prompts,answers,values=False):
    ids=torch.tensor([tokens(p+r) for p,r in zip(prompts,answers)])
    logits=model(ids[:,:-1])[0][:,-2:,48:58]
    target=ids[:,-2:]-48;logps=logits.log_softmax(-1)
    selected=logps.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    return selected, -(logps.exp()*logps).sum(-1).mean()


def policy_objective(new,old,advantages,algorithm,clip=.2):
    # Ratios are formed from *length-normalized* log likelihood for GSPO.
    if algorithm=="reinforce":return -(new.sum(-1)*advantages).mean()
    if algorithm=="gspo":ratio=(new-old).mean(-1).exp();adv=advantages
    else:ratio=(new-old).exp();adv=advantages[:,None]
    return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()


def reinforcement(a):
    torch.manual_seed(a.seed);data=arithmetic();model=TinyTransformer(Config(width=32,layers=1,position="rope"))
    fit(model,data["train"],a.steps,a.seed,lr=.005)
    reference=copy.deepcopy(model); reference.eval()
    for p in reference.parameters():p.requires_grad_(False)
    prompts=[p for p,r in data[a.split]];expected=[r for p,r in data[a.split]]
    before=digit_rollout(model,prompts,True)
    # Tiny prompt-level value baseline for the terminal return used by PPO.
    critic=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=.01)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007);rng=random.Random(a.seed);history=[]
    for step in range(a.steps):
        chosen=[rng.choice(data["train"]) for _ in range(4)]; ps=[p for p,r in chosen for _ in range(4)]; targets=[r for p,r in chosen for _ in range(4)]
        answers=digit_rollout(model,ps)
        rewards=torch.tensor([float(ans==target) if a.reward=="exact" else float(ans.startswith("0")) for ans,target in zip(answers,targets)])
        with torch.no_grad():old,_=digit_logps(model,ps,answers);ref,_=digit_logps(reference,ps,answers)
        if a.experiment in ("grpo","gspo"):
            groups=rewards.view(-1,4);advantages=((groups-groups.mean(1,keepdim=True))/(groups.std(1,keepdim=True,unbiased=False)+1e-4)).flatten()
        elif a.experiment=="ppo":
            features=torch.tensor([[int(p[0])/9,int(p[2])/9,0.] for p in ps]);predicted=critic(features).squeeze(-1)
            advantages=(rewards-predicted.detach())
            value_loss=F.mse_loss(predicted,rewards);critic_optimizer.zero_grad();value_loss.backward();critic_optimizer.step()
        else:advantages=rewards-rewards.mean()
        # REINFORCE is on-policy: update once, then collect a fresh rollout.
        # Clipped objectives permit the two declared surrogate optimization steps.
        for _ in range(policy_updates_per_rollout(a.experiment)):
            new,entropy=digit_logps(model,ps,answers)
            # Nonnegative sample-based KL estimator; expectation is taken under rollout policy.
            delta=ref-new;kl=(delta.exp()-delta-1).mean()
            loss=policy_objective(new,old,advantages,a.experiment)+a.beta*kl-.005*entropy
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%10==0 or step+1==a.steps:
            history.append({"step":step+1,"loss":float(loss.detach()),"training_reward":float(rewards.mean()),"reward_standard_deviation":float(rewards.std(unbiased=False))})
    after=digit_rollout(model,prompts,True);export_model(a.out,model,"3+4=",task="arithmetic")
    rows=[{"prompt":p,"expected":r,"before":b,"after":z} for p,r,b,z in zip(prompts,expected,before,after)]
    plt.figure(figsize=(7,4));plt.plot([h['step'] for h in history],[h['training_reward'] for h in history]);plt.xlabel('Policy update');plt.ylabel('Training reward');plt.ylim(-.05,1.05);plt.tight_layout();plt.savefig(a.out/'rewards.png',dpi=130);plt.close()
    return {"metrics":{"heldout_exact_match_before":sum(x==y for x,y in zip(before,expected))/len(expected),"heldout_exact_match_after":sum(x==y for x,y in zip(after,expected))/len(expected),"reward_rule":a.reward,"predictions":rows,"policy_updates_per_rollout":policy_updates_per_rollout(a.experiment)},
            "history":history,"artifacts":["rewards.png"],"reload_verified":True,
            "limitations":"Two-digit constrained autoregressive policy; same fixed-length arithmetic simulator. PPO uses a prompt-state value baseline and terminal return, with no generalized advantage estimation. This is a PPO clipping/critic teaching implementation, not a production trainer. Group algorithms use four samples per prompt. Zero-variance groups provide zero normalized reward advantage. The deliberately flawed zero-prefix reward does not measure arithmetic correctness."}


def pass_at_k(n,c,k):
    if not 0<=c<=n or not 1<=k<=n:raise ValueError("Require 0 <= correct <= samples and 1 <= k <= samples.")
    return 1. if n-c<k else 1-math.comb(n-c,k)/math.comb(n,k)


def wilson(correct,total):
    if total<1:raise ValueError("Need observations.")
    p=correct/total;z=1.96;den=1+z*z/total
    center=(p+z*z/(2*total))/den;spread=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-spread),min(1.,center+spread)]


def evaluation(a):
    torch.manual_seed(a.seed);data=arithmetic();model=TinyTransformer(Config(layers=1,position="rope"));fit(model,data["train"],a.steps,a.seed)
    rows=[]
    for p,r in data[a.split]:
        samples=digit_rollout(model,[p]*8);c=sum(s==r for s in samples)
        rows.append({"prompt":p,"expected":r,"samples":samples,"correct":c,"pass_at_1":pass_at_k(8,c,1),"pass_at_4":pass_at_k(8,c,4)})
    greedy=digit_rollout(model,[p for p,r in data[a.split]],True);correct=sum(x==r for x,(p,r) in zip(greedy,data[a.split]))
    export_model(a.out,model,"3+4=",task="arithmetic")
    (a.out/'samples.json').write_text(json.dumps(rows,indent=2))
    # This loss is over the full byte vocabulary; sampling above constrains digits.
    loss=loss_on(model,data[a.split])
    metrics={"greedy_exact_match":correct/len(rows),"greedy_wilson_95_interval":wilson(correct,len(rows)),"mean_pass_at_1":float(np.mean([r['pass_at_1'] for r in rows])),"mean_pass_at_4":float(np.mean([r['pass_at_4'] for r in rows])),"byte_perplexity":math.exp(loss),"examples":len(rows),"samples_per_example":8,"sampling_support":"decimal digits only","raw_samples_file":"samples.json"}
    return {"metrics":metrics,"reload_verified":True,"limitations":"Original arithmetic mini-suite, not AIME, MMLU, GPQA, HLE, or SWE-bench. Eight stochastic samples per problem; Wilson interval applies to greedy problem-level outcomes, not correlated samples. Full-vocabulary byte perplexity and constrained-generation accuracy measure different distributions."}


def policy_updates_per_rollout(algorithm):
    return 1 if algorithm=='reinforce' else 2


def data_card(a):
    card={'origin':'Original authored or simulated teaching data','license':'Included for course exercises',
          'real_world_validation':'Not performed; these scores do not establish deployment reliability.'}
    if a.experiment=='serving':
        return card|{'dataset':'Event-sentence training data and fixed local timing prompts',
            'split_policy':'Systems timing probe; no held-out quality evaluation',
            'selected_split':'systems_timing_probe','training_rows':len(notes()['train']),
            'timing_prompts':['service api: ','service api event start: service queue event restore: '],
            'requests':12,'protocol':'Two prompts, cached/uncached, three warmed serial requests per combination'}
    if a.experiment=='judging':
        from modern_evaluation import CASE_BANKS
        return card|{'dataset':'Authored factual-comparison cases, not human preference collection',
            'split_policy':'Disjoint question banks; two answer-order presentations per case',
            'selected_split':a.split,'case_ids':{key:[f'{key}:{i}' for i in range(len(rows))] for key,rows in CASE_BANKS.items()},
            'independent_cases':len(CASE_BANKS[a.split]),'presentations':2*len(CASE_BANKS[a.split])}
    if a.experiment=='reward':
        rows=reward_pairs();dataset='Authored style-preference pairs; no factuality labels';policy='Distinct train, validation, and test response-pair fixtures'
    elif a.experiment in {'tuning','preference'}:
        rows=tickets();dataset='Fictional support tickets';policy='Ticket-template split; shared categories remain'
    elif a.experiment in {'reinforce','ppo','grpo','gspo','evaluation'}:
        rows=arithmetic();dataset='Two-operand addition simulator';policy='Operand-pair split; shared arithmetic rule remains'
    else:
        rows=notes();dataset='Fictional event sentences';policy='Whole-sentence split; shared rules and vocabulary remain'
    return card|{'dataset':dataset,'split_policy':policy,'selected_split':a.split,
        'partition_sizes':{key:len(value) for key,value in rows.items()},
        'row_ids':{key:[hashlib.sha256(json.dumps(row).encode()).hexdigest()[:16] for row in value] for key,value in rows.items()}}


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("experiment",choices=EXPERIMENTS);p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=100);p.add_argument("--seed",type=int,default=42);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--split",choices=("validation","test"),default=None);p.add_argument("--beta",type=float,default=.1)
    p.add_argument("--reward",choices=("exact","zero-prefix"),default="exact");p.add_argument("--judge",choices=("fixture","model"),default="fixture")
    return p


def run(a):
    if a.experiment=='serving':
        if a.split is not None:raise ValueError('--split does not apply to fixed systems timing probes.')
    else:a.split=a.split or 'validation'
    if not 1<=a.steps<=1000 or not 1<=a.threads<=16 or not 0<=a.beta<=2 or not 0<=a.seed<2**32:raise ValueError("Use 1–1000 steps, 1–16 threads, beta in [0,2], and an unsigned 32-bit seed.")
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    a.out=Path(a.output)
    if a.out.exists() and any(a.out.iterdir()):raise ValueError("Choose a new output folder; existing evidence will not be overwritten.")
    a.out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    if a.experiment in {"attention","components","experts","research"}:result=model_experiment(a)
    elif a.experiment=="tuning":result=tuning(a)
    elif a.experiment=="preference":result=preference(a)
    elif a.experiment=="reward":result=reward(a)
    elif a.experiment in {"reinforce","ppo","grpo","gspo"}:result=reinforcement(a)
    elif a.experiment=="evaluation":result=evaluation(a)
    elif a.experiment=="judging":
        from modern_evaluation import judging
        result=judging(a)
    elif a.experiment=="serving":
        from modern_serving import serving_experiment
        result=serving_experiment(a)
    provenance=data_card(a)
    report={"schema":"superlab.report.v1","experiment":a.experiment,"seed":a.seed,"split":provenance['selected_split'],"data":provenance,"settings":{k:v for k,v in vars(a).items() if k not in {'out','output'}},
            "runtime":{"torch":torch.__version__,"device":"cpu","seconds":time.perf_counter()-started},"artifacts":[],**result}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (a.out/'data-card.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(report,indent=2,allow_nan=False));return report


if __name__=="__main__":run(parser().parse_args())
