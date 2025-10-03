import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
import os
import time
from typing import Dict, Any

import __init__
from model.model import Etude, EtudeHFConfig
from config import PretrainConfig, DataConfig, TokenizerConfig
from data_utils import StreamingDataset



def train() -> None:
    p_cfg = PretrainConfig()
    d_cfg = DataConfig()
    t_cfg = TokenizerConfig()

    os.makedirs(p_cfg.pretrain_model_dir, exist_ok=True)

    print("加载分词器...")
    if not os.path.isdir(t_cfg.TOKENIZER_DIR):
        print(f"找不到分词器目录 {t_cfg.TOKENIZER_DIR}")
        return

    tokenizer = AutoTokenizer.from_pretrained(t_cfg.TOKENIZER_DIR)

    print("创建模型配置...")
    m_cfg = EtudeHFConfig(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    print(f"分词器加载成功，词表大小: {m_cfg.vocab_size}")

    effective_batch_size = p_cfg.batch_size * p_cfg.accumulation_steps
    print("--- 预训练配置 ---")
    print(f"设备: {p_cfg.device}")
    print(f"有效批量大小: {effective_batch_size}")
    print(f"模型将保存至: {p_cfg.pretrain_model_dir}")
    print("--------------------")

    dataset = StreamingDataset(d_cfg, tokenizer, mode='pretrain')
    dataloader = DataLoader(
        dataset,
        batch_size=p_cfg.batch_size,
        num_workers=p_cfg.num_workers,
        pin_memory=True
    )

    model: Etude
    if os.path.exists(os.path.join(p_cfg.pretrain_model_dir, "pytorch_model.bin")):
        print(f"[恢复训练] 从HF模型目录 {p_cfg.pretrain_model_dir} 加载模型")
        model = Etude.from_pretrained(p_cfg.pretrain_model_dir).to(p_cfg.device)
    else:
        print("未找到现有模型，将从头初始化。")
        model = Etude(m_cfg).to(p_cfg.device)

    print("模型初始化完成。")

    optimizer = AdamW(model.parameters(), lr=p_cfg.lr)

    scaler = torch.amp.GradScaler(enabled=('cuda' in p_cfg.device))

    global_step = 0
    start_epoch = 0
    if os.path.exists(p_cfg.checkpoint_file):
        print(f"[恢复训练状态] 从 {p_cfg.checkpoint_file} 加载...")
        checkpoint: Dict[str, Any] = torch.load(p_cfg.checkpoint_file, map_location=p_cfg.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and ('cuda' in p_cfg.device):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"成功恢复训练状态到 Step {global_step}, Epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, p_cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{p_cfg.epochs} ===")
        epoch_start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(p_cfg.device, non_blocking=True), y.to(p_cfg.device, non_blocking=True)


            device_type = p_cfg.device.split(':')[0]
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=('cuda' in p_cfg.device)):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                if loss is not None:
                    loss = loss / p_cfg.accumulation_steps

            if loss is not None:
                scaler.scale(loss).backward()

            if (batch_idx + 1) % p_cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step > 0 and global_step % p_cfg.print_every == 0 and loss is not None:
                    print(f"Step {global_step} | loss: {loss.item() * p_cfg.accumulation_steps:.4f}")

                if global_step > 0 and global_step % p_cfg.save_every == 0:
                    print(f"\n[保存] Step {global_step}: 保存检查点和模型...")
                    model.save_pretrained(p_cfg.pretrain_model_dir, safe_serialization=False)
                    tokenizer.save_pretrained(p_cfg.pretrain_model_dir)
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, p_cfg.checkpoint_file)
                    print(f"检查点和模型已保存至: {p_cfg.pretrain_model_dir}\n")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成，耗时: {epoch_duration:.2f} 秒")

    print("\n预训练完成，保存最终模型...")
    model.save_pretrained(p_cfg.pretrain_model_dir, safe_serialization=False)
    tokenizer.save_pretrained(p_cfg.pretrain_model_dir)
    print(f"最终模型和分词器已保存至: {p_cfg.pretrain_model_dir}")

if __name__ == "__main__":
    train()