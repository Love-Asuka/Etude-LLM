import os
import json
from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

class TokenizerConfig:
    FILES_PATTERNS = [
        "training_data/xml/pretrain_hq.jsonl",
        "training_data/big_text/sft_mini_512.jsonl"
    ]
    
    VOCAB_SIZE = 20000

    SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "[UNK]", "<|im_end|>", "<|im_start|>"]
    
    TOKENIZER_FILE = "weight/tokenizer.json" 

def universal_text_iterator(files_patterns: list):
    for pattern in files_patterns:
        files = glob(pattern)
        if not files:
            print(f"警告: 在 '{pattern}' 路径下未找到任何文件。")
            continue
        
        print(f"在 '{pattern}' 中找到 {len(files)} 个文件，开始提取文本...")
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "text" in data:
                            yield data["text"]
                        elif "conversations" in data:
                            for message in data["conversations"]:
                                if "content" in message:
                                    yield message["content"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

def train_the_one_tokenizer():
    cfg = TokenizerConfig()
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()

    trainer = BpeTrainer(vocab_size=cfg.VOCAB_SIZE, special_tokens=cfg.SPECIAL_TOKENS)

    print("开始训练")
    data_iterator = universal_text_iterator(cfg.FILES_PATTERNS)
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    print("训练结束")

    output_dir = os.path.dirname(cfg.TOKENIZER_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"已经保存到: {cfg.TOKENIZER_FILE}")
    print(f"它的词表大小是: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":

    train_the_one_tokenizer()

