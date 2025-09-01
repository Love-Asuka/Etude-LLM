import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import tiktoken
from model import Etude, GPTConfig


class SFTDataset(Dataset):
    def __init__(self, path, block_size=2048):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.bos_token = self.enc.encode("<|startoftext|>", allowed_special={"<|startoftext|>"})[0]
        self.sep_token = self.enc.encode("\n", allowed_special={"\n"})[0] 
        self.encoded_data = [] 
        self.max_lines = 1500  # 每个文件读取的最大行数
        raw_conversations = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    conversations = data.get("conversations", [])
                    if conversations:
                        raw_conversations.append(conversations)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        for conv in raw_conversations:
            full_encoded = [self.bos_token]
            for turn in conv:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role and content:
                    role_encoded = self.enc.encode(role + ": ")
                    content_encoded = self.enc.encode(content)
                    full_encoded.extend(role_encoded + content_encoded + [self.sep_token])
            
            full_encoded.append(self.eos_token)

            for i in range(0, len(full_encoded), self.block_size):
                chunk = full_encoded[i:i+self.block_size+1]
                if len(chunk) < self.block_size + 1:
                    chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
                self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """带warmup的余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_steps, training_steps, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.training_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def train_sft(model, optimizer, scheduler, train_loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"SFT训练第 {epoch+1}/{total_epochs} 轮", unit="batch")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, targets=y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / len(train_loader)


def evaluate_sft(model, val_loader, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def main2():
    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    model = Etude(config).to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    

    warmup_steps = 500
    
    total_train_steps = 10000  # 可以根据需要调整，自己算一下
    
    # 带warmup的余弦退火调度器
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_train_steps
    )
    
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"使用设备: {device}")

    # 检查点恢复
    checkpoint_path = "weight\weight_sft\sft_model_latest.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从检查点恢复训练，从第 {start_epoch} 轮开始")

    # 训练数据设置
    folder_path = "jsonl_sft"
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    random.shuffle(jsonl_files)  # 随机
    
    num_jsonls_to_train = min(800, len(jsonl_files))  # 训练文件数量
    epochs_per_file = 1  # 单个文件训练的轮数
    total_epochs = num_jsonls_to_train * epochs_per_file
    save_interval = 100  # 多少epoch保存一次
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, total_epochs):
        file_idx = epoch % num_jsonls_to_train
        file_path = os.path.join(folder_path, jsonl_files[file_idx])
        
        # 加载数据集并随机划分训练集和验证集
        full_dataset = SFTDataset(file_path)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        
        train_loss = train_sft(model, optimizer, scheduler, train_loader, device, epoch, total_epochs)
        val_loss = evaluate_sft(model, val_loader, device)
        
        print(f'Epoch: {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # 保存
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # 最佳模型保存
        if (epoch + 1) % save_interval == 0:
            torch.save(checkpoint, f'weight\weight_sft\sft_model_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'weight\weight_sft\sft_model_best.pt')
            print(f"新的最佳模型，验证损失: {val_loss:.4f}")


if __name__ == "__main__":
    import math
    main2()