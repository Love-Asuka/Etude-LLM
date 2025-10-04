import torch
import json
import os
import shutil
import argparse
from collections import OrderedDict

def calculate_intermediate_size(n_embd: int, multiple_of: int = 256) -> int:
    hidden_dim = int(n_embd * 4 * (2 / 3))
    intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return intermediate_size

def convert_etude_to_llama(input_dir: str, output_dir: str):
    print("--- Etude to Llama 伪装脚本 (V2 - 修正版) ---")
    print(f"源模型目录: {input_dir}")
    print(f"目标目录: {output_dir}\n")

    # ... (步骤 1 和 2 保持不变) ...
    print("步骤 1: 加载原始 Etude 模型文件...")
    etude_config_path = os.path.join(input_dir, "config.json")
    etude_model_path = os.path.join(input_dir, "pytorch_model.bin")
    if not os.path.exists(etude_config_path) or not os.path.exists(etude_model_path):
        print(f"[错误] 在 '{input_dir}' 中未找到 'config.json' 或 'pytorch_model.bin'。")
        return
    with open(etude_config_path, 'r', encoding='utf-8') as f:
        etude_config = json.load(f)
    etude_state_dict = torch.load(etude_model_path, map_location="cpu")
    print("原始文件加载成功。\n")

    print("步骤 2: 转换模型配置 (config.json)...")
    n_embd = etude_config["n_embd"]
    n_layer = etude_config["n_layer"]
    n_head = etude_config["n_head"]
    vocab_size = etude_config["vocab_size"]
    tokenizer_config_path = os.path.join(input_dir, "tokenizer_config.json")
    bos_token_id = 1
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tok_conf = json.load(f)
            if "bos_token" in tok_conf and isinstance(tok_conf["bos_token"], dict):
                 bos_token_id = tok_conf["bos_token"].get("id", 1)
            elif "bos_token_id" in tok_conf:
                 bos_token_id = tok_conf["bos_token_id"]
    llama_config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": n_embd, "num_hidden_layers": n_layer,
        "num_attention_heads": n_head, "vocab_size": vocab_size,
        "intermediate_size": calculate_intermediate_size(n_embd),
        "rms_norm_eps": 1e-06, "max_position_embeddings": 4096,
        "bos_token_id": bos_token_id, "eos_token_id": etude_config["eos_token_id"],
        "pad_token_id": etude_config["pad_token_id"], "torch_dtype": "float32",
        "tie_word_embeddings": etude_config.get("tie_word_embeddings", True),
    }
    print("配置转换完成。\n")

    # ... (步骤 3 保持不变) ...
    print("步骤 3: 转换模型权重 (state dict)...")
    llama_state_dict = OrderedDict()
    llama_state_dict["model.embed_tokens.weight"] = etude_state_dict["token_embedding.weight"]
    for i in range(n_layer):
        qkv_weight = etude_state_dict[f"blocks.{i}.att.qkv_proj.weight"]
        q_proj, k_proj, v_proj = torch.split(qkv_weight, n_embd, dim=0)
        llama_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = q_proj
        llama_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = k_proj
        llama_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = v_proj
        llama_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = etude_state_dict[f"blocks.{i}.att.out_proj.weight"]
        llama_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = etude_state_dict[f"blocks.{i}.ffn.net.w1.weight"]
        llama_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = etude_state_dict[f"blocks.{i}.ffn.net.w3.weight"]
        llama_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = etude_state_dict[f"blocks.{i}.ffn.net.w2.weight"]
        llama_state_dict[f"model.layers.{i}.input_layernorm.weight"] = etude_state_dict[f"blocks.{i}.ln1.weight"]
        llama_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = etude_state_dict[f"blocks.{i}.ln2.weight"]
    llama_state_dict["model.norm.weight"] = etude_state_dict["ln_f.weight"]
    llama_state_dict["lm_head.weight"] = etude_state_dict["lm_head.weight"]
    print("权重转换完成。\n")
    
    # --- 4. 验证与保存 (已修正) ---
    print("步骤 4: 验证并保存转换后的模型...")
    total_etude_params = sum(p.numel() for p in etude_state_dict.values())
    total_llama_params = sum(p.numel() for p in llama_state_dict.values())
    if total_etude_params != total_llama_params:
        print(f"[警告] 参数量不匹配! Etude: {total_etude_params}, Llama: {total_llama_params}")
    else:
        print(f"参数量验证通过: {total_llama_params/1e6:.2f}M parameters.")

    os.makedirs(output_dir, exist_ok=True)
    
    # ↓↓↓ --- 这是核心修正 --- ↓↓↓
    print("正在复制所有辅助文件 (tokenizer configs, templates, etc.)...")
    for filename in os.listdir(input_dir):
        # 我们要生成新的权重和配置，所以跳过它们，复制其他所有文件
        if filename not in ["pytorch_model.bin", "config.json"]:
            source_file = os.path.join(input_dir, filename)
            dest_file = os.path.join(output_dir, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
    print("辅助文件复制完成。")
    # ↑↑↑ --- 修正结束 --- ↑↑↑

    # 保存新的 Llama 配置文件 (这将覆盖任何已复制的 config.json)
    with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(llama_config, f, indent=2, ensure_ascii=False)

    # 保存新的 Llama 权重文件
    torch.save(llama_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    print("\n--- 转换成功！ ---")
    print(f"Llama 兼容模型已保存至: {output_dir}")

# ... (主函数入口 __main__ 保持不变) ...
if __name__ == "__main__":
    input_path = "weight/etude_sft_model/"
    output_path = "weight/etude_sft_model_llama_format/"

    # 直接用定义好的路径调用函数
    print(f"注意：正在使用脚本内硬编码的路径！")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    convert_etude_to_llama(input_path, output_path)