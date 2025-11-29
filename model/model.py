import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


class EtudeHFConfig(PretrainedConfig):
    model_type = "etude"

    def __init__(
        self,
        vocab_size: int = 16384,
        n_layer: int = 6,
        n_head: int = 4,
        n_embd: int = 768,
        dropout: float = 0.1,

        tie_word_embeddings: bool = True,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        use_cache: bool = True,

        **kwargs
    ):

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.head_size = self.n_embd // self.n_head
        self.use_cache = use_cache

        self.num_hidden_layers = n_layer    
        self.num_attention_heads = n_head    
        self.hidden_size = n_embd    

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        if seq_len + seq_len_offset > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len + seq_len_offset, device=x.device, dtype=x.dtype)
        
        cos = self.cos_cached[seq_len_offset : seq_len + seq_len_offset]
        sin = self.sin_cached[seq_len_offset : seq_len + seq_len_offset]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.n_embd = config.n_embd
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_size, max_position_embeddings=4096)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, past_key_value: Optional[Cache] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None, layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        seq_len_offset = 0
        if past_key_value is not None:
            seq_len_offset = past_key_value.get_seq_length(layer_idx)
        
        cos, sin = self.rope(q, seq_len_offset=seq_len_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, cache_kwargs={"sin": sin, "cos": cos})

        present_key_value = None 
        
        is_causal = x.size(1) > 1
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), present_key_value

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class FeedForward(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        hidden_dim = int(config.n_embd * 4 * (2 / 3))
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.net = SwiGLU(dim=config.n_embd, hidden_dim=hidden_dim, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)
        self.ln2 = nn.RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Cache] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None, layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x_norm = self.ln1(x)
        x_att, present_kv = self.att(x_norm, past_key_value, use_cache, attention_mask, layer_idx=layer_idx)
        x = residual + x_att

        residual = x
        x_norm = self.ln2(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn
        return x, present_kv

class Etude(PreTrainedModel, GenerationMixin):
    config_class = EtudeHFConfig

    def __init__(self, config: EtudeHFConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        past_length = 0
        if past_key_values is not None:
             past_length = past_key_values.get_seq_length()
        
        if past_length > 0:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        return model_inputs

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.token_embedding = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = self.token_embedding(input_ids)
        
        if use_cache and past_key_values is None:
             past_key_values = DynamicCache()

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.view(input_ids.shape[0], 1, 1, -1).to(dtype=torch.bool)

        for i, block in enumerate(self.blocks):
            x, _ = block(x, past_key_values, use_cache, attention_mask, layer_idx=i)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + (None,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )