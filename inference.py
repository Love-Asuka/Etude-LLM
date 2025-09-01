import torch
import json
import os
from model import Etude
from train import GPTConfig
import tiktoken
import random


torch.manual_seed(random.randint(0,5000))
config = GPTConfig()

model = Etude(config)
device = "cuda"
model = model.to(device)


checkpoint_path = "weight\weight_train\model_epoch_latest.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已从检查点 {checkpoint_path} 加载")
else:
    raise FileNotFoundError(f"检查点文件 {checkpoint_path} 不存在")


model.eval()


enc = tiktoken.get_encoding("gpt2")

def generate_text(prompt, max_new_tokens=50):

    input_ids = enc.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    

    generated_ids = model.generate(input_tensor, max_new_tokens)

    generated_text = enc.decode(generated_ids[0].tolist())
    
    return generated_text

def chat():
    print("输入 '退出' 结束对话。")
    context = ""
    
    while True:
        user_input = input("你: ")
        
        if user_input.lower() == "退出":
            print("对话结束。")
            break
        context += f"用户: {user_input}\n助手: "
        response = generate_text(context, max_new_tokens=100)
        assistant_response = response.split("助手: ")[-1].split("\n")[0]
        print(f"助手: {assistant_response}")
        context += f"{assistant_response}\n"

if __name__ == "__main__":
    chat()
