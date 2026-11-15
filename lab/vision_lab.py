"""Optional model-card-based SmolVLM inference on a local image; separate extension."""
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from lab import resolve

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--prompt", default="Describe the image briefly.")
args = parser.parse_args()
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
revision = resolve(model_id)
torch.set_num_threads(4)
processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision,
    torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True).eval()
picture = Image.open(args.image).convert("RGB")
messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text", "text":args.prompt}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[picture], return_tensors="pt")
with torch.inference_mode():
    result = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(result[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
