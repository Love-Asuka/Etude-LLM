import torch
import json
import itertools
from glob import glob
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerFast
from typing import List, Dict, Tuple, Iterator, Optional
from collections import deque

from config import DataConfig

class DialogueFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_start_token_id is None or self.im_end_token_id is None:
            raise ValueError("Tokenizer缺少必要的特殊token: <|im_start|> or <|im_end|>")
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    def format_conversation(self, conversations: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        full_token_ids = []
        for msg in conversations:
            role, content = msg.get("role", ""), msg.get("content", "")
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>"
            full_token_ids.extend(self.tokenizer.encode(prefix + content + suffix, add_special_tokens=False))

        if len(full_token_ids) < 2: return None
        full_token_ids = full_token_ids[:self.block_size + 1]
        input_ids = full_token_ids[:-1]
        labels = full_token_ids[1:]

        current_pos = 0
        for msg in conversations:
            if current_pos >= len(labels): break
            role, content = msg.get("role", ""), msg.get("content", "")
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>"
            prefix_len = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            content_len = len(self.tokenizer.encode(content, add_special_tokens=False))
            turn_len = prefix_len + content_len + len(self.tokenizer.encode(suffix, add_special_tokens=False))
            
            if role != 'assistant':
                for i in range(current_pos, min(current_pos + turn_len, len(labels))): labels[i] = -100
            else:
                for i in range(current_pos, min(current_pos + prefix_len, len(labels))): labels[i] = -100
                for i in range(current_pos + prefix_len + content_len, min(current_pos + turn_len, len(labels))): labels[i] = -100
            current_pos += turn_len

        pad_len = self.block_size - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long))

def universal_text_iterator(files_patterns: List[str]) -> Iterator[str]:
    for pattern in files_patterns:
        for file_path in glob(pattern):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        yield from extract_text_from_json_recursive(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

def extract_text_from_json_recursive(data) -> Iterator[str]:
    if isinstance(data, str): yield data
    elif isinstance(data, dict):
        for value in data.values(): yield from extract_text_from_json_recursive(value)
    elif isinstance(data, list):
        for item in data: yield from extract_text_from_json_recursive(item)

class StreamingDataset(IterableDataset):
    def __init__(self, data_cfg: DataConfig, tokenizer: PreTrainedTokenizerFast, mode: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = data_cfg.block_size
        self.mode = mode

        if self.mode == 'pretrain':
            self.file_path = data_cfg.pretrain_file
            self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        elif self.mode == 'sft':
            self.file_path = data_cfg.sft_file
            self.formatter = DialogueFormatter(tokenizer, data_cfg.block_size)
        else:
            raise ValueError(f"不支持的模式: {mode}.")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            worker_f = itertools.islice(f, worker_id, None, num_workers)
            
            if self.mode == 'sft':
                yield from self._sft_iterator(worker_f)
            else: 
                yield from self._pretrain_iterator(worker_f)

    def _sft_iterator(self, file_iterator: Iterator[str]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for line in file_iterator:
            try:
                data = json.loads(line.strip())
                if "conversations" in data:
                    formatted_sample = self.formatter.format_conversation(data["conversations"])
                    if formatted_sample: yield formatted_sample
            except (json.JSONDecodeError, KeyError):
                continue
    
    def _pretrain_iterator(self, file_iterator: Iterator[str]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:

        token_buffer: List[int] = []
        for line in file_iterator:
            try:
                text = json.loads(line.strip()).get("text", "")
                if text:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(ids + [self.eos_token_id])

                    while len(token_buffer) >= self.block_size + 1:

                        x_tokens = token_buffer[0:self.block_size]
                        y_tokens = token_buffer[1:self.block_size + 1]
                        yield (
                            torch.tensor(x_tokens, dtype=torch.long),
                            torch.tensor(y_tokens, dtype=torch.long)
                        )
                        token_buffer = token_buffer[self.block_size:]
            except json.JSONDecodeError:
                continue