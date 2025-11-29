import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

import __init__
from model.model import Etude, EtudeHFConfig 
from train.config import SFTConfig

from transformers import AutoConfig
AutoConfig.register("etude", EtudeHFConfig)
from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
_AutoModelForCausalLM.register(EtudeHFConfig, Etude)


def run_inference():
    sft_cfg = SFTConfig()
    device = sft_cfg.device

    print("加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(sft_cfg.sft_model_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            sft_cfg.sft_model_dir,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map="auto" if device == 'cuda' else None,
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        print(f"请确保 '{sft_cfg.sft_model_dir}' 包含所有必要的文件。")
        return
        
    print("模型加载成功。")

    conversation_history: List[Dict[str, str]] = []
    print("\n=== Etude Chat ===")
    print("输入 'exit' 或 'quit' 结束对话。\n")
    
    while True:
        try:
            user_input = input("用户: ").strip()
            if not user_input: continue
            if user_input.lower() in {"exit", "quit"}: break
            
            conversation_history.append({"role": "user", "content": user_input})
            
            reply = generate_reply(
                model, 
                tokenizer, 
                conversation_history, 
                device
            )
            
            conversation_history.append({"role": "assistant", "content": reply})
            print(f"Etude: {reply}\n")
            
        except KeyboardInterrupt:
            break
            
    print("\nEtude: 再见。")


@torch.no_grad()
def generate_reply(
    model: Etude, 
    tokenizer: AutoTokenizer, 
    conversation_history: List[Dict[str, str]], 
    device: str,
    max_new_tokens: int = 512, 
    temperature: float = 0.7, 
    top_p: float = 0.9,
) -> str:
    prompt_ids = tokenizer.apply_chat_template(
        conversation_history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True 
    )
    

    reply_ids = generated_ids[0][len(prompt_ids[0]):]
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

    return reply

if __name__ == "__main__":
    run_inference()