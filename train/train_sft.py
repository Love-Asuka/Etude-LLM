import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
import os
import time
from typing import Dict, Any

import __init__
from model.model import Etude, EtudeHFConfig
from config import SFTConfig, DataConfig, TokenizerConfig
from data_utils import StreamingDataset

def sft_train() -> None:
    sft_cfg = SFTConfig()
    d_cfg = DataConfig()
    t_cfg = TokenizerConfig()

    os.makedirs(sft_cfg.sft_model_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(t_cfg.TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        
    effective_batch_size = sft_cfg.batch_size * sft_cfg.accumulation_steps
    print("--- 指令微调 SFT 配置 ---")
    print(f"设备: {sft_cfg.device}")
    print(f"有效批量大小: {effective_batch_size}")
    print(f"SFT模型将保存至: {sft_cfg.sft_model_dir}")
    print("-------------------------")

    dataset = StreamingDataset(d_cfg, tokenizer, mode='sft')
    dataloader = DataLoader(
        dataset, 
        batch_size=sft_cfg.batch_size, 
        num_workers=sft_cfg.num_workers, 
        pin_memory=True
    )

    model: Etude
    
    sft_model_path = os.path.join(sft_cfg.sft_model_dir, "pytorch_model.bin")
    pretrain_model_path = os.path.join(sft_cfg.pretrain_model_dir, "pytorch_model.bin")

    if os.path.exists(sft_model_path):
        print(f"[恢复SFT训练] 从HF模型目录 {sft_cfg.sft_model_dir} 加载模型")
        model = Etude.from_pretrained(sft_cfg.sft_model_dir).to(sft_cfg.device)
    elif os.path.exists(pretrain_model_path):
        print(f"未找到SFT模型，从预训练模型目录 {sft_cfg.pretrain_model_dir} 加载")
        model = Etude.from_pretrained(sft_cfg.pretrain_model_dir).to(sft_cfg.device)
    else:
        print("[错误] 未找到保存点或预训练权重：退出。")
        return

    print("模型初始化完成。")

    optimizer = AdamW(model.parameters(), lr=sft_cfg.lr)
    scaler = torch.amp.GradScaler(enabled=('cuda' in sft_cfg.device))

    global_step = 0
    start_epoch = 0
    if os.path.exists(sft_cfg.checkpoint_file):
        print(f"[恢复训练状态] 从 {sft_cfg.checkpoint_file} 加载...")
        checkpoint: Dict[str, Any] = torch.load(sft_cfg.checkpoint_file, map_location=sft_cfg.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and ('cuda' in sft_cfg.device):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"成功恢复训练状态到 Step {global_step}, Epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, sft_cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{sft_cfg.epochs} ===")
        epoch_start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(sft_cfg.device, non_blocking=True)
            y = y.to(sft_cfg.device, non_blocking=True)

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            attention_mask = (x != pad_id).long().to(sft_cfg.device, non_blocking=True)

            device_type = sft_cfg.device.split(':')[0]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=('cuda' in sft_cfg.device)):

                outputs = model(
                    input_ids=x, 
                    labels=y, 
                    attention_mask=attention_mask, 
                    use_cache=False 
                )
                loss = outputs.loss

                if loss is not None:
                    loss = loss / sft_cfg.accumulation_steps

            if loss is not None:
                scaler.scale(loss).backward()

            if (batch_idx + 1) % sft_cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step > 0 and global_step % sft_cfg.print_every == 0 and loss is not None:
                    print(f"Step {global_step} | loss: {loss.item() * sft_cfg.accumulation_steps:.4f}")

                if global_step > 0 and global_step % sft_cfg.save_every == 0:
                    print(f"\n[保存] Step {global_step}: 保存检查点和模型...")
                    model.save_pretrained(sft_cfg.sft_model_dir, safe_serialization=False)
                    tokenizer.save_pretrained(sft_cfg.sft_model_dir)
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, sft_cfg.checkpoint_file)
                    print(f"检查点和模型已保存至: {sft_cfg.sft_model_dir}\n")
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成，耗时: {epoch_duration:.2f} 秒")

    print("\nSFT完成")
    model.save_pretrained(sft_cfg.sft_model_dir, safe_serialization=False)
    tokenizer.save_pretrained(sft_cfg.sft_model_dir)
    print(f"模型和分词器已完整保存至: {sft_cfg.sft_model_dir}")

if __name__ == "__main__":
    sft_train()