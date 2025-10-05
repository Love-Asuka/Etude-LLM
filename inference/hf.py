import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

MODEL_PATH = "weight/etude_sft_model_llama_format/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

def load_model_and_tokenizer():
    print(f"使用的设备: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def stream_reply(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(DEVICE)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    parts = []
    print("\n模型: ", end="", flush=True)
    with torch.no_grad():
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        for text in streamer:
            parts.append(text)
            print(text, end="", flush=True)
        thread.join()
    print("")
    return "".join(parts).strip()


def chat():
    model, tokenizer = load_model_and_tokenizer()
    print("\n--- Etude ---")
    print("输入 'quit' 或 'exit' 退出。输入 'clear' 清空对话历史。")

    messages = []
    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                break
            if user_input.lower() == "clear":
                messages.clear()
                print("对话历史已清空。")
                continue

            messages.append({"role": "user", "content": user_input})
            response = stream_reply(model, tokenizer, messages)
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat()