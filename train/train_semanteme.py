import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
import torch.amp as amp
import json
from tokenizers import Tokenizer 
import os
import time

import __init__  

from model.model import Etude, EtudeConfig

class TrainingConfig:

    model_config = EtudeConfig(
        batch_size=20,       
    )

    train_file = "training_data/xml/pretrain_hq.jsonl"
    tokenizer_file = "weight/tokenizer.json" 
    weight_file = "weight/weight_semanteme/etude_model_optimized.pt"
    checkpoint_file = "weight/weight_semanteme/etude_checkpoint_optimized.pt"

    block_size = 512
    epochs = 3
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    num_workers = 16          
    accumulation_steps = 1   
    

    print_every = 100       
    save_every = 10000      


    effective_batch_size = model_config.batch_size * accumulation_steps


class EfficientStreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, block_size, text_buffer_size=2000): 
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer 
        self.block_size = block_size
        self.text_buffer_size = text_buffer_size
        self.eos_token = self.tokenizer.token_to_id("<|endoftext|>") 

    def __iter__(self):
        text_buffer = []
        token_buffer = []

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    text = json.loads(line.strip())["text"]
                    text_buffer.append(text)
                except (json.JSONDecodeError, KeyError):
                    continue
                
                if len(text_buffer) >= self.text_buffer_size:
                    encoded_lines = self.tokenizer.encode_batch(text_buffer)
                    text_buffer = []
                    
                    for encoded in encoded_lines:
                        token_buffer.extend(encoded.ids + [self.eos_token]) # <--- 获取ID列表

                    processed_offset = 0
                    while processed_offset + self.block_size + 1 <= len(token_buffer):
                        start = processed_offset
                        end = start + self.block_size + 1
                        
                        x = torch.tensor(token_buffer[start:end-1], dtype=torch.long)
                        y = torch.tensor(token_buffer[start+1:end], dtype=torch.long)
                        yield x, y

                        processed_offset += self.block_size
                    
                    token_buffer = token_buffer[processed_offset:]

        if text_buffer:
            encoded_lines = self.tokenizer.encode_batch(text_buffer)
            for encoded in encoded_lines:
                token_buffer.extend(encoded.ids + [self.eos_token])
        
        processed_offset = 0
        while processed_offset + self.block_size + 1 <= len(token_buffer):
            start = processed_offset
            end = start + self.block_size + 1

            x = torch.tensor(token_buffer[start:end-1], dtype=torch.long)
            y = torch.tensor(token_buffer[start+1:end], dtype=torch.long)
            yield x, y

            processed_offset += self.block_size


def train():
    cfg = TrainingConfig()
    os.makedirs(os.path.dirname(cfg.weight_file), exist_ok=True)


    print("加载分词器...")
    if not os.path.exists(cfg.tokenizer_file):
        print(f"错误：找不到分词器文件 {cfg.tokenizer_file}！先去运行 train_tokenizer.py！")
        return
    tokenizer = Tokenizer.from_file(cfg.tokenizer_file)
    cfg.model_config.vocab_size = tokenizer.get_vocab_size()
    cfg.model_config.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"分词器加载成功，词表大小: {cfg.model_config.vocab_size}")

    print("--- 配置信息 ---")
    print(f"设备: {cfg.device}")
    print(f"微批量大小 (Micro Batch Size): {cfg.model_config.batch_size}")
    print(f"梯度累积步数 (Accumulation Steps): {cfg.accumulation_steps}")
    print(f"有效批量大小 (Effective Batch Size): {cfg.effective_batch_size}")
    print(f"DataLoader Workers: {cfg.num_workers}")
    print("------------------")

    dataset = EfficientStreamingDataset(
        cfg.train_file, 
        tokenizer=tokenizer, 
        block_size=cfg.block_size
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.model_config.batch_size, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    print("正在初始化 Etude 模型...")
    model = Etude(cfg.model_config).to(cfg.device) 
    print("模型初始化完成。")
    

    
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scaler = amp.GradScaler(device=cfg.device, enabled=(cfg.device == 'cuda'))

    global_step = 0
    start_epoch = 0
    if os.path.exists(cfg.checkpoint_file):
        print(f"[恢复训练] 从 {cfg.checkpoint_file} 加载...")
        checkpoint = torch.load(cfg.checkpoint_file, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", scaler.state_dict()))
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"成功恢复到 Step {global_step}, Epoch {start_epoch}")
    else:
        print("未找到 checkpoint")


    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
        import sys; sys.stdout.flush()
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            
            with amp.autocast(device_type=cfg.device, enabled=(cfg.device == 'cuda'), dtype=torch.float16):
                logits, loss = model(x, y)
                loss = loss / cfg.accumulation_steps
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step > 0 and global_step % cfg.print_every == 0:
                    print(f"Step {global_step} | loss: {loss.item() * cfg.accumulation_steps:.4f}")
                    sys.stdout.flush()

                if global_step > 0 and global_step % cfg.save_every == 0:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, cfg.checkpoint_file)
                    torch.save(model.state_dict(), cfg.weight_file)
                    print(f"\n[保存] Step {global_step}:")
                    print(f"  -> 检查点: {cfg.checkpoint_file}")
                    print(f"  -> 权重: {cfg.weight_file}\n")
                    sys.stdout.flush()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成，耗时: {epoch_duration:.2f} 秒")

    print("\n训练完成，正在保存最终模型和检查点...")
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "epoch": cfg.epochs - 1,
    }
    torch.save(final_checkpoint, cfg.checkpoint_file)
    torch.save(model.state_dict(), cfg.weight_file)
    print(f"最终模型权重已保存到: {cfg.weight_file}")
    print(f"最终检查点已保存到: {cfg.checkpoint_file}")


if __name__ == "__main__":
    train()