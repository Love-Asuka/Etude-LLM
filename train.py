from model import Etude, MyDataset 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model import GPTConfig
import os

def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"训练第 {epoch+1}/{total_epochs} 轮", unit="batch")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"损失": f"{loss.item():.4f}"})
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

def main1():
    config = GPTConfig()
    model = Etude(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # 使用 ReduceLROnPlateau 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")

    folder_path = "json"  # 路径
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    num_jsonls_to_train = 300  # 文件数量
    epochs_per_file = 1  #轮数
    total_epochs = num_jsonls_to_train * epochs_per_file 

    start_epoch = 0
    checkpoint_path = "weight\weight_train\model_epoch_latest.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从检查点恢复训练，从第 {start_epoch} 轮开始")

    save_interval = 100  # 保存间隔

    for epoch in range(start_epoch, total_epochs):
        file_idx = (epoch // epochs_per_file) % num_jsonls_to_train
        file_path = os.path.join(folder_path, jsonl_files[file_idx % len(jsonl_files)])
        train_dataset = MyDataset(file_path)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch, total_epochs)
        val_loss = eval(model, val_loader, device)
        print(f'轮次: {epoch+1}, 训练损失: {train_loss/len(train_loader):.4f}, 验证损失: {val_loss/len(val_loader):.4f}')

        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }

        # 更新调度器
        scheduler.step(avg_val_loss)

        if (epoch + 1) % save_interval == 0:#保存判断
            torch.save(checkpoint, f'weight/model_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main1()
