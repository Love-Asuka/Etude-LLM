import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
import torch.amp as amp
import torch.nn.functional as F
import json
import os
from tokenizers import Tokenizer 
import __init__  

from model.model import Etude, EtudeConfig

class SFTConfig:
    model_config = EtudeConfig(
        batch_size=16,        
        expert_number=2,
        top_k=2,
    )
    sft_file = "training_data/big_text/sft_mini_512.jsonl"
    block_size = 512
    epochs = 3
    lr = 3e-5  
    
    tokenizer_file = "weight/tokenizer.json" 
    pretrain_weight = "weight/weight_semanteme/etude_model_optimized.pt"
    sft_weight = "weight/weight_full_sft/etude_sft.pt"
    checkpoint_file = "weight/weight_full_sft/etude_sft_checkpoint.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16        
    accumulation_steps = 1  
    
    print_every = 100
    save_every = 5000

    effective_batch_size = model_config.batch_size * accumulation_steps


class EfficientStreamingSFTDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, block_size, text_buffer_size=1000):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer 
        self.block_size = block_size
        self.text_buffer_size = text_buffer_size
        self.eos_token = self.tokenizer.token_to_id("<|endoftext|>") 

    def __iter__(self):
        token_buffer = []
        mask_buffer = []
        data_buffer = []

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "conversations" in data:
                        data_buffer.append(data["conversations"])
                except (json.JSONDecodeError, KeyError):
                    continue
                
                if len(data_buffer) >= self.text_buffer_size:
                    new_tokens, new_masks = self._process_data_buffer(data_buffer)
                    token_buffer.extend(new_tokens)
                    mask_buffer.extend(new_masks)
                    data_buffer = []

                    processed_offset = 0
                    while processed_offset + self.block_size + 1 <= len(token_buffer):
                        start = processed_offset
                        end = start + self.block_size + 1
                        
                        x = torch.tensor(token_buffer[start:end-1], dtype=torch.long)
                        y = torch.tensor(token_buffer[start+1:end], dtype=torch.long)
                        m = torch.tensor(mask_buffer[start+1:end], dtype=torch.float32)
                        yield x, y, m

                        processed_offset += self.block_size
                    
                    token_buffer = token_buffer[processed_offset:]
                    mask_buffer = mask_buffer[processed_offset:]

        if data_buffer:
            new_tokens, new_masks = self._process_data_buffer(data_buffer)
            token_buffer.extend(new_tokens)
            mask_buffer.extend(new_masks)
        
        processed_offset = 0
        while processed_offset + self.block_size + 1 <= len(token_buffer):
            start = processed_offset
            end = start + self.block_size + 1

            x = torch.tensor(token_buffer[start:end-1], dtype=torch.long)
            y = torch.tensor(token_buffer[start+1:end], dtype=torch.long)
            m = torch.tensor(mask_buffer[start+1:end], dtype=torch.float32)
            yield x, y, m
            
            processed_offset += self.block_size

    def _process_data_buffer(self, data_buffer):
        all_tokens = []
        all_masks = []
        for conversations in data_buffer:
            tokens = []
            masks = []
            for msg in conversations:
                role = msg["role"]
                content = msg["content"]
                token_ids = self.tokenizer.encode(content).ids 
                tokens.extend(token_ids + [self.eos_token])
                if role == "user":
                    masks.extend([0] * (len(token_ids) + 1))
                elif role == "assistant":
                    masks.extend([1] * (len(token_ids) + 1))
            all_tokens.extend(tokens)
            all_masks.extend(masks)
        return all_tokens, all_masks


def sft_train():
    cfg = SFTConfig()
    os.makedirs(os.path.dirname(cfg.sft_weight), exist_ok=True)

    print("加载分词器...")
    if not os.path.exists(cfg.tokenizer_file):
        print(f"错误：找不到分词器文件 {cfg.tokenizer_file}！")
        return
    tokenizer = Tokenizer.from_file(cfg.tokenizer_file)
    cfg.model_config.vocab_size = tokenizer.get_vocab_size()
    cfg.model_config.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"分词器加载成功，词表大小: {cfg.model_config.vocab_size}")

    print("--- SFT 配置信息 ---")
    print(f"设备: {cfg.device}")
    print(f"微批量大小 (Micro Batch Size): {cfg.model_config.batch_size}")
    print(f"梯度累积步数 (Accumulation Steps): {cfg.accumulation_steps}")
    print(f"有效批量大小 (Effective Batch Size): {cfg.effective_batch_size}")
    print("------------------")

    dataset = EfficientStreamingSFTDataset(
        cfg.sft_file, 
        tokenizer=tokenizer, 
        block_size=cfg.block_size
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.model_config.batch_size, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    model = Etude(cfg.model_config).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scaler = amp.GradScaler(device=cfg.device, enabled=(cfg.device == 'cuda'))


    global_step = 0
    start_epoch = 0
    if os.path.exists(cfg.checkpoint_file):
        print(f"[恢复SFT训练] 从 {cfg.checkpoint_file} 加载")
        checkpoint = torch.load(cfg.checkpoint_file, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        print(f"成功恢复到 Step {global_step}, Epoch {start_epoch}")
    elif os.path.exists(cfg.pretrain_weight):
        print(f"未找到SFT检查点，从预训练权重 {cfg.pretrain_weight} 加载。")
        state_dict = torch.load(cfg.pretrain_weight, map_location=cfg.device)
        model.load_state_dict(state_dict, strict=False) 
        print("预训练权重加载成功，将开始新的SFT。")
    else:
        print("警告：未找到SFT检查点和预训练权重，将从头开始训练。")
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
        for batch_idx, (x, y, mask) in enumerate(dataloader):
            x, y, mask = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True), mask.to(cfg.device, non_blocking=True)

            with amp.autocast(device_type=cfg.device, enabled=(cfg.device == 'cuda'), dtype=torch.float16):
                logits, _ = model(x)
                loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
                loss = (loss_all * mask.view(-1)).sum() / (mask.sum() + 1e-8)
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
                    import sys; sys.stdout.flush()

                if global_step > 0 and global_step % cfg.save_every == 0:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, cfg.checkpoint_file)
                    torch.save(model.state_dict(), cfg.sft_weight)
                    print(f"\n[保存] Step {global_step}:")
                    print(f"  -> 检查点: {cfg.checkpoint_file}")
                    print(f"  -> 权重: {cfg.sft_weight}\n")
                    sys.stdout.flush()

    print("\nSFT完成，正在保存最终模型和检查点...")
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "epoch": cfg.epochs - 1,
    }
    torch.save(final_checkpoint, cfg.checkpoint_file)
    torch.save(model.state_dict(), cfg.sft_weight)
    print(f"最终SFT权重已保存到 {cfg.sft_weight}")
    print(f"最终检查点已保存到 {cfg.checkpoint_file}")


if __name__ == "__main__":
    sft_train()