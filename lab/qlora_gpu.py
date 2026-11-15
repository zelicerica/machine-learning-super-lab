"""Optional genuine NormalFloat4 QLoRA extension. Requires a compatible CUDA GPU.

This hardware-specific extension is separate from the verified CPU curriculum.
Use a new output directory. Read course/QLoRA-and-Cloud.md before installing.
"""
import argparse
import gc
import json
import random
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=30);a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('A compatible CUDA graphics processor is required. The CPU packed-four-bit lesson is a different implementation; see the cloud guide.')
    if not 1<=a.steps<=1000:raise SystemExit('Use 1–1000 steps.')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory.')
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
    import lab
    model_id='HuggingFaceTB/SmolLM2-135M-Instruct';revision=lab.resolve(model_id);torch.manual_seed(42)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    def load_base():return AutoModelForCausalLM.from_pretrained(model_id,revision=revision,quantization_config=config,device_map={'':0},trust_remote_code=False,use_safetensors=True)
    model=prepare_model_for_kbit_training(load_base())
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,target_modules='all-linear',lora_dropout=0.0))
    model.config.use_cache=False
    train=[lab.encode_example(tokenizer,row,256) for row in lab.read_rows('train')]
    valid=[lab.encode_example(tokenizer,row,256) for row in lab.read_rows('validation')]
    before=lab.validation_loss(model,valid,tokenizer,'cuda');model.train()
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.0002);rng=random.Random(42)
    torch.cuda.reset_peak_memory_stats();history=[]
    for step in range(a.steps):
        example=lab.collate([rng.choice(train)],tokenizer.pad_token_id,'cuda');loss=model(**example).loss
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; inspect hardware, precision and data.')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        history.append({'step':step+1,'loss':float(loss.detach())});print(history[-1],flush=True)
    model.eval();after=lab.validation_loss(model,valid,tokenizer,'cuda');peak=torch.cuda.max_memory_allocated()
    probe=lab.collate([valid[0]],tokenizer.pad_token_id,'cuda')
    with torch.no_grad():original=model(**probe).logits.cpu()
    out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out);tokenizer.save_pretrained(out)
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
    model=PeftModel.from_pretrained(load_base(),out).eval()
    with torch.no_grad():reloaded=model(**probe).logits.cpu()
    error=float((original-reloaded).abs().max());close=torch.allclose(original,reloaded,atol=.05,rtol=.01)
    report={'schema':'superlab.report.v1','experiment':'qlora-nf4','split':'validation','model':model_id,'revision':revision,'history':history,
            'metrics':{'loss_before':before,'loss_after':after,'peak_allocated_cuda_bytes':peak,'reload_max_logit_error':error},'reload_verified':close,
            'limitations':'Tiny synthetic ticket dataset; CUDA allocated-memory measurement excludes driver and some library allocations. Actual hardware execution is required; this script was not GPU-validated in the CPU release environment.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if not close:raise RuntimeError('Saved adapter reload did not meet the documented numeric tolerance; inspect the saved report.')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
