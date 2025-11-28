import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class TokenizerConfig:
    FILES_PATTERNS: List[str] = field(default_factory=lambda: [
        "training_data/pretrain/pretrain_hq.jsonl",
        "training_data/sft/sft_mini_512.jsonl",
        "training_data/dpo/dpo.jsonl"
    ])
    CACHE_FILE: str = "training_data/tokenizer_cache.txt"
    VOCAB_SIZE: int = 16384
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: [
        "<|endoftext|>", 
        "<|im_start|>", 
        "<|im_end|>"
    ])
    TOKENIZER_DIR: str = "weight/tokenizer/"
    TOKENIZER_FILE: str = "weight/tokenizer/tokenizer.json"

@dataclass
class DataConfig:
    pretrain_file: str = "training_data/pretrain/pretrain_hq.jsonl"
    sft_file: str = "training_data/sft/sft_mini_512.jsonl"
    block_size: int = 512
    text_buffer_size: int = 4000

@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 16
    epochs: int = 3
    accumulation_steps: int = 4
    print_every: int = 100
    save_every: int = 5000

@dataclass
class PretrainConfig(TrainingConfig):
    batch_size: int = 16
    lr: float = 3e-4
    pretrain_model_dir: str = "weight/etude_pretrained_model/"
    checkpoint_file: str = "weight/etude_pretrained_model/pretrain_checkpoint.pt"

@dataclass
class SFTConfig(TrainingConfig):
    batch_size: int = 8
    lr: float = 3e-5
    pretrain_model_dir: str = "weight/etude_pretrained_model/"
    sft_model_dir: str = "weight/etude_sft_model/"
    checkpoint_file: str = "weight/etude_sft_model/sft_checkpoint.pt"