import os
import json
import argparse
from glob import glob
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer

import __init__
from config import TokenizerConfig
from data_utils import universal_text_iterator

ADVANCED_CHATML_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
    "{% else %}"
        "{% set loop_messages = messages %}"
        "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant...') }}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def verify_config(cfg: TokenizerConfig) -> bool:

    print("--- 正在验证 Tokenizer 配置 ---")
    print(f"数据源 patterns: {cfg.FILES_PATTERNS}")
    print(f"词表大小: {cfg.VOCAB_SIZE}")
    print(f"特殊 Tokens: {cfg.SPECIAL_TOKENS}")
    print(f"输出目录: {cfg.TOKENIZER_DIR}")
    
    has_data = False
    for pattern in cfg.FILES_PATTERNS:
        files = glob(pattern)
        if files:
            print(f"  [成功] 在 '{pattern}' 路径下找到 {len(files)} 个文件。")
            has_data = True
        else:
            print(f"  [警告] 在 '{pattern}' 路径下未找到任何文件。")
    
    if not has_data:
        print("\n[错误] 未找到任何训练数据文件！请检查 config.py 中的 FILES_PATTERNS。")
        return False
        
    return True

def train_the_one_tokenizer(cfg: TokenizerConfig):
    print("\n--- 开始训练 Tokenizer ---")
    
    print(f"创建缓存文件: {cfg.CACHE_FILE}")
    os.makedirs(os.path.dirname(cfg.CACHE_FILE), exist_ok=True)
    with open(cfg.CACHE_FILE, 'w', encoding='utf-8') as f:
        for text in universal_text_iterator(cfg.FILES_PATTERNS):
            if text and text.strip(): f.write(text.strip() + '\n')


    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.VOCAB_SIZE, 
        special_tokens=cfg.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print("开始训练Tokenizer...")
    tokenizer.train([cfg.CACHE_FILE], trainer=trainer)
    print("Tokenizer训练完成。")


    os.makedirs(cfg.TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"核心分词器已保存至: {cfg.TOKENIZER_FILE}")
    
    print("正在创建Hugging Face兼容配置文件...")
    endoftext_token, im_start_token, im_end_token = cfg.SPECIAL_TOKENS
    
    special_tokens_map = {
        "bos_token": im_start_token, "eos_token": im_end_token,
        "pad_token": endoftext_token, "unk_token": endoftext_token,
        "additional_special_tokens": [im_start_token, im_end_token]
    }
    with open(os.path.join(cfg.TOKENIZER_DIR, "special_tokens_map.json"), 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    config_data = {
        "model_max_length": 4096, "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": ADVANCED_CHATML_TEMPLATE, "add_prefix_space": False,
        "clean_up_tokenization_spaces": False, **special_tokens_map
    }
    with open(os.path.join(cfg.TOKENIZER_DIR, "tokenizer_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"所有Tokenizer配置文件已保存至: {cfg.TOKENIZER_DIR}")


    os.remove(cfg.CACHE_FILE)
    print("已删除缓存文件。")
    print("\nTokenizer训练和配置全部完成！")

def verify_tokenizer(cfg: TokenizerConfig):
    print("\n开始验证")
    
    if not os.path.isdir(cfg.TOKENIZER_DIR):
        print(f"[错误] Tokenizer目录 '{cfg.TOKENIZER_DIR}' 不存在")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_DIR)
        print("使用 AutoTokenizer 加载成功！")

        print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
        assert tokenizer.pad_token == "<|endoftext|>"
        assert tokenizer.eos_token == "<|im_end|>"
        assert tokenizer.bos_token == "<|im_start|>"
        print("特殊token映射正确！")

        print("\n正在测试Chat Template (带system prompt):")
        messages_with_system = [{"role": "system", "content": "You are Etude, a helpful AI."}, {"role": "user", "content": 'Hello!'}]
        prompt1 = tokenizer.apply_chat_template(messages_with_system, tokenize=False, add_generation_prompt=True)
        print(prompt1)
        assert "You are Etude, a helpful AI." in prompt1
        assert prompt1.strip().endswith("<|im_start|>assistant")
        print("带system prompt测试通过！")
        
        print("\n正在测试Chat Template (无system prompt):")
        messages_no_system = [{"role": "user", "content": 'Hi there!'}]
        prompt2 = tokenizer.apply_chat_template(messages_no_system, tokenize=False, add_generation_prompt=True)
        print(prompt2)
        assert "You are a helpful assistant." in prompt2
        assert prompt2.strip().endswith("<|im_start|>assistant")
        print("无system prompt（默认）测试通过！")
        print("\n--- Tokenizer 验证成功！ ---")

    except Exception as e:
        print(f"\n[错误] 验证失败: {e}")

#其实我完全可以把验证删掉，因为我跑通了，但是我懒得删
#而且万一哪天我改了配置文件又忘了改回来了呢

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etude-LLM Tokenizer 训练和验证脚本")
    parser.add_argument(
        "action",
        type=str,
        choices=["config", "train", "verify", "all"],
        default="all",
        nargs="?", 
        help=(
            "执行的操作: "
            "'config' - 仅验证配置文件并退出; "
            "'train' - 验证配置后进行训练; "
            "'verify' - 验证已存在的tokenizer; "
            "'all' - (默认) 验证配置、训练并进行最终验证。"
        )
    )
    args = parser.parse_args()

    config = TokenizerConfig()

    if args.action == 'config':
        verify_config(config)
    
    elif args.action == 'train':
        if verify_config(config):
            train_the_one_tokenizer(config)
            
    elif args.action == 'verify':
        verify_tokenizer(config)
        
    elif args.action == 'all':
        if verify_config(config):
            train_the_one_tokenizer(config)
            verify_tokenizer(config)