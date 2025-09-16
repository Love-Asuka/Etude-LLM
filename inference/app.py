import gradio as gr
import torch
from tokenizers import Tokenizer
import __init__
from model.model import Etude, EtudeConfig

import time



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

tokenizer_file = "weight/tokenizer.json"
weight_file = "weight/weight_full_sft/etude_sft.pt"

print("正在加载分词器...")
try:
    tokenizer = Tokenizer.from_file(tokenizer_file)
except FileNotFoundError:
    print(f"错误：找不到分词器文件 {tokenizer_file}")
    exit()
print(f"分词器加载成功，词表大小: {tokenizer.get_vocab_size()}")

config = EtudeConfig()
config.vocab_size = tokenizer.get_vocab_size()
config.eos_token_id = tokenizer.token_to_id("<|endoftext|>")

if config.eos_token_id is None:
    print("错误：分词器里没有找到 '<|endoftext|>'")
    exit()

print("正在加载模型...")
model = Etude(config).to(device)
try:
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.eval()
    print("模型加载成功。")
except Exception as e:
    print(f"加载模型权重时发生错误: {e}")
    exit()


@torch.no_grad()
def generate_reply_streaming(user_input, history, max_new_tokens=1024, temperature=0.7, top_k=40):
    
    model.eval()

   
    history.append({"role": "user", "content": user_input})
    
    history.append({"role": "assistant", "content": ""})

    
    yield None, history

    # 准备完整的 prompt 输入给模型
    # 注意：我们只需要 history[:-1] 的内容，因为最后一个是空的占位符
    prompt_tokens = []
    for message in history[:-1]:
        prompt_tokens.extend(tokenizer.encode(message['content']).ids)
        prompt_tokens.append(config.eos_token_id)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    
    output_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        
        if next_id.item() == config.eos_token_id:
            break

       
        output_ids.append(next_id.item())
        
        
        assistant_reply_so_far = tokenizer.decode(output_ids)

        
        history[-1]['content'] = assistant_reply_so_far

        
        yield None, history

        
        input_ids = torch.cat([input_ids, next_id], dim=1)



with gr.Blocks(theme=gr.themes.Soft(), title="Etude") as demo:
    gr.Markdown("# Etude")
    gr.Markdown("在下方的文本框中输入你的问题，然后按回车。")
    
    chatbot = gr.Chatbot(label="对话窗口", height=500, type='messages')
    
    msg = gr.Textbox(label="你的输入", placeholder="在这里输入...", show_label=False)
    
    clear = gr.ClearButton([msg, chatbot], value="清空对话")

    
    msg.submit(generate_reply_streaming, [msg, chatbot], [msg, chatbot])

# 启动界面
if __name__ == "__main__":
    print("正在启动 Gradio 界面...")
    demo.launch(share=True)