import torch
from tokenizers import Tokenizer
import __init__
from model.model import Etude, EtudeConfig


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_file = "weight/tokenizer.json"
weight_file = "weight/weight_full_sft/etude_sft.pt"

print("加载分词器...")
try:
    tokenizer = Tokenizer.from_file(tokenizer_file)
except FileNotFoundError:
    print(f"错误：找不到分词器 {tokenizer_file}")
    exit()
print(f"分词器加载成功，词表大小: {tokenizer.get_vocab_size()}")

config = EtudeConfig()
config.vocab_size = tokenizer.get_vocab_size()
config.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
if config.eos_token_id is None:
    print("错误：分词器里没有找到 '<|endoftext|>'")
    exit()


print("正在加载模型...")
import sys
sys.stdout.flush()
model = Etude(config).to(device)
try:
    model.load_state_dict(torch.load(weight_file, map_location=device))
except Exception as e:
    print(f"错误: {e}")
    exit()
model.eval()
print("模型加载成功。")


conversation_history = []

@torch.no_grad()
def generate_reply(model, tokenizer, user_input, max_new_tokens=512, temperature=0.8, top_k=40):
    model.eval()

    conversation_history.append({"role": "user", "content": user_input})

    prompt_tokens = []
    for msg in conversation_history:
        prompt_tokens.extend(tokenizer.encode(msg['content']).ids)
        prompt_tokens.append(config.eos_token_id)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    input_len = input_ids.shape[1]


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
            
        input_ids = torch.cat([input_ids, next_id], dim=1)

    output_ids = input_ids[0, input_len:].tolist()
    assistant_reply = tokenizer.decode(output_ids).strip()

    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply


if __name__ == "__main__":
    print("\n--- Etude 对话界面 ---")
    print("输入 'exit' 或 'quit' 结束对话。\n")
    while True:
            prompt = input().strip()
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                print("\nEtude: 再见。")
                break
            
            reply = generate_reply(model, tokenizer, prompt)
            print(f"Etude: {reply}\n")
