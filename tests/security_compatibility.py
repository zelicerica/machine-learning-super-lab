"""Offline CPU regression: tiny random Llama + Low-Rank Adaptation (LoRA).

This checks package APIs, optimizer updates, safetensors saving, adapter reload
and new-input generation. It is not pretrained inference or a quality benchmark.
Repeatable command from project root:
python tests/security_compatibility.py --output lab/outputs/security-compat-01
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    base=LlamaForCausalLM(config)
    base.save_pretrained(out/'base',safe_serialization=True)
    original={name:p.detach().clone() for name,p in base.named_parameters()}
    model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=4,lora_alpha=8,lora_dropout=0,target_modules=['q_proj','v_proj']))
    train=torch.randint(3,64,(4,16),generator=torch.Generator().manual_seed(101))
    evaluation=torch.randint(3,64,(2,16),generator=torch.Generator().manual_seed(202))
    new=torch.tensor([[1,8,17,23]])
    def loss():
        model.eval()
        with torch.no_grad():return float(model(input_ids=evaluation,labels=evaluation).loss)
    before=loss();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    losses=[]
    for i in range(4):
        model.train();optimizer.zero_grad(set_to_none=True)
        value=model(input_ids=train[i:i+1],labels=train[i:i+1]).loss
        if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
        value.backward();optimizer.step();losses.append(float(value.detach()))
    after=loss();model.eval()
    with torch.no_grad():expected=model(new).logits
    model.save_pretrained(out/'adapter',safe_serialization=True)
    restored=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(out/'base',local_files_only=True,trust_remote_code=False,use_safetensors=True),out/'adapter',local_files_only=True).eval()
    with torch.no_grad():actual=restored(new).logits
    torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-6)
    with torch.no_grad():generated=restored.generate(input_ids=new,max_new_tokens=3,do_sample=False,pad_token_id=0)
    # PEFT wraps modules and changes names; the underlying frozen base remains identical.
    frozen=all(torch.equal(p.detach(),original[name.replace('.base_layer','')]) for name,p in model.get_base_model().named_parameters() if 'lora_' not in name)
    if not frozen:raise RuntimeError('Frozen base weights changed')
    contract={'scope':'tiny model initialized randomly; LoRA adaptation API regression','input':'integer token IDs 0..63, batch x sequence, context <=128','target':'next token, labels shifted by model','split':'independent training seed 101 and evaluation seed 202; new input is separate','model':'one-layer Llama, 32 hidden units; no pretrained weights or tokenizer','deployment':'local CPU CLI only','production_gaps':'synthetic random token data; no useful language quality or production claim'}
    report={'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate','pillow']},'training_losses':losses,'independent_eval_loss_before':before,'independent_eval_loss_after':after,'frozen_base_unchanged':frozen,'reload_verified':True,'max_logit_error':float((expected-actual).abs().max()),'new_input':new.tolist(),'generated_ids':generated.tolist(),'vision_auto_class_imported':AutoModelForImageTextToText.__name__,'limits':'No pretrained language or image checkpoint, GPU, remote service, browser visual or production-quality test.'}
    (out/'input-contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'artifact-hashes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
