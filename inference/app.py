import gradio as gr
import torch
from tokenizers import Tokenizer
import __init__
from model.model import Etude, EtudeConfig

import time

# -------------------------------------------------------------------
# 初始化部分 (和原来一样)
# -------------------------------------------------------------------

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
    """
    一个生成器函数，逐 token 生成并流式输出回复。
    """
    model.eval()

    # 1. 将用户的最新输入添加到历史记录中
    history.append({"role": "user", "content": user_input})
    # 2. 为即将生成的助手回复添加一个空的占位符
    history.append({"role": "assistant", "content": ""})

    # 立即 yield 一次，这样用户的输入会立刻显示在聊天窗口中
    yield None, history

    # 3. 准备完整的 prompt 输入给模型
    # 注意：我们只需要 history[:-1] 的内容，因为最后一个是空的占位符
    prompt_tokens = []
    for message in history[:-1]:
        prompt_tokens.extend(tokenizer.encode(message['content']).ids)
        prompt_tokens.append(config.eos_token_id)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # 4. 【关键改动】逐 token 生成并在循环中 yield
    output_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # 如果生成了结束符，就提前停止
        if next_id.item() == config.eos_token_id:
            break

        # 将新生成的 token id 添加到输出列表中
        output_ids.append(next_id.item())
        
        # 将当前所有已生成的 token 解码成字符串
        assistant_reply_so_far = tokenizer.decode(output_ids)

        # 更新历史记录中最后一个（也就是助手）的消息内容
        history[-1]['content'] = assistant_reply_so_far

        # 【关键】yield 更新后的历史记录，Gradio 会自动刷新 chatbot 界面
        yield None, history

        # 将新生成的 token 拼接到下一次的输入中，准备生成再下一个 token
        input_ids = torch.cat([input_ids, next_id], dim=1)



with gr.Blocks(theme=gr.themes.Soft(), title="Etude") as demo:
    gr.Markdown("# Etude")
    gr.Markdown("在下方的文本框中输入你的问题，然后按回车。")
    
    chatbot = gr.Chatbot(label="对话窗口", height=500, type='messages')
    
    msg = gr.Textbox(label="你的输入", placeholder="在这里输入...", show_label=False)
    
    clear = gr.ClearButton([msg, chatbot], value="清空对话")

    # 将提交动作绑定到新的流式生成函数
    msg.submit(generate_reply_streaming, [msg, chatbot], [msg, chatbot])

# 启动界面
if __name__ == "__main__":
    print("正在启动 Gradio 界面...")
    demo.launch(share=True)