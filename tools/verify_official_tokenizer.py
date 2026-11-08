#!/usr/bin/env python3
"""Compare standalone assembly CLI IDs with the official tokenizer."""
import os
os.environ.setdefault('HF_HOME','/tmp/plamo-hf')
from pathlib import Path
import random
import subprocess
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def main():
 t=AutoTokenizer.from_pretrained(ROOT/'models/plamo',trust_remote_code=True)
 cases=['','Hello','こんにちは😀','a  b<|plamo:bos|>日本語','aaaaa','日日日日','\n\n\n\n','𠮷野家','<|plamo:unknown|>','<|plamo:|>','a\uee00b']
 for text,src,dst in [('Hello.','en','ja'),('こんにちは。','ja','en')]:
  cases.append(t.apply_chat_template([{'role':'user','content':text}],source_lang=src,target_lang=dst,tokenize=False,add_generation_prompt=True))
 rng=random.Random(4)
 for _ in range(20):cases.append(''.join(rng.choice('abc日東京 \n😀')*rng.randrange(1,7) for _ in range(10)))
 for text in cases:
  result=subprocess.run([str(ROOT/'build/plamo'),text],cwd=ROOT,check=True,capture_output=True,text=True)
  actual=list(map(int,result.stdout.split()));expected=t.encode(text,add_special_tokens=False)
  assert actual==expected,(text,actual,expected)
 print(f'official tokenizer vs standalone assembly CLI: PASS ({len(cases)} cases)')
if __name__=='__main__':main()
