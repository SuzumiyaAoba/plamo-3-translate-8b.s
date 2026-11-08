#!/usr/bin/env python3
"""Official FP32 single-prompt reference, loading one decoder layer at a time."""
import os
os.environ.setdefault('HF_HOME','/tmp/plamo-hf')
import sys,json,importlib.util,argparse
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'models/plamo'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('text',nargs='?',default='Hello');parser.add_argument('--source');parser.add_argument('--target');parser.add_argument('--append-id',type=int,action='append',default=[]);args=parser.parse_args()
 torch.set_num_threads(4)
 spec=importlib.util.spec_from_file_location('official_plamo',MODEL/'modeling_plamo.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 config=mod.Plamo3Config(**json.loads((MODEL/'config.json').read_text()))
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 handles={name:safe_open(str(MODEL/name),framework='pt') for name in set(index.values())}
 def tensor(name):return handles[index[name]].get_tensor(name).float()
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
 text=args.text
 if args.target:
  text=tokenizer.apply_chat_template([{'role':'user','content':text}],source_lang=args.source,target_lang=args.target,tokenize=False,add_generation_prompt=True)
 ids=tokenizer.encode(text,add_special_tokens=False)+args.append_id
 print('input',ids,flush=True)
 with torch.inference_mode():
  emb=handles[index['model.embed_tokens.weight']].get_slice('model.embed_tokens.weight')
  x=torch.stack([emb[i:i+1].float()[0] for i in ids])[None]
  for i in range(config.num_hidden_layers):
   prefix=f'model.layers.layers.{i}.'
   with torch.device('meta'):layer=mod.Plamo3DecoderLayer(config,i)
   layer.load_state_dict({k[len(prefix):]:tensor(k) for k in index if k.startswith(prefix)},assign=True)
   layer.mixer.rotary_emb.device=torch.device('cpu');layer.mixer.rotary_emb._compute_buffers()
   length=x.shape[1]
   mask=torch.zeros((1,1,length,length)).masked_fill(torch.triu(torch.ones(length,length,dtype=torch.bool),diagonal=1),float('-inf'))
   x=layer(x,attention_mask=mask)[0]
   print('layer',i,'rms',x.square().mean().sqrt().item(),flush=True)
   x[0,-1].contiguous().numpy().tofile(ROOT/f'build/ref_layer_{i}.bin')
   del layer
  x=mod._plamo3_rms_norm(x,tensor('model.norm.weight'),config.rms_norm_eps)
  head=handles[index['lm_head.weight']].get_slice('lm_head.weight');best=(-float('inf'),-1)
  for start in range(0,config.vocab_size,4096):
   logits=torch.mv(head[start:start+4096].float(),x[0,-1]);val,pos=logits.max(0)
   if val.item()>best[0]:best=(val.item(),start+pos.item())
  print('next',best,'text',repr(tokenizer.decode([best[1]])),flush=True)
if __name__=='__main__':main()
