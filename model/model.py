import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


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

        **kwargs
    ):

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


        self.head_size = self.n_embd // self.n_head


        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        return (self._norm(x.float()) * self.weight).to(output_dtype)

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
        # x shape: (B, H, T, D_h)
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
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D_h)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D_h)
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

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, D_h)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, D_h)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, H, T, D_h)

        seq_len_offset = 0
        if past_key_value is not None:
            seq_len_offset = past_key_value[0].shape[2]
        
        cos, sin = self.rope(q, seq_len_offset=seq_len_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None
        

        is_causal = attention_mask is None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
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

        self.net = SwiGLU(
            dim=config.n_embd,
            hidden_dim=hidden_dim,
            dropout=config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ln1 = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x_norm = self.ln1(x)
        x_att, present_kv = self.att(x_norm, past_key_value, use_cache, attention_mask)
        x = residual + x_att

        residual = x
        x_norm = self.ln2(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn
        
        return x, present_kv

class Etude(PreTrainedModel):
    config_class = EtudeHFConfig

    def __init__(self, config: EtudeHFConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.post_init()
        
    def generate(self, input_ids, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9, 
                 eos_token_id=None, pad_token_id=None, **kwargs):

        batch_size = input_ids.shape[0]
        device = input_ids.device
        

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
            
        generated_tokens = input_ids.clone()
        

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        

        for _ in range(max_new_tokens):

            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_tokens,
                    use_cache=True,
                    **kwargs
                )
                
  
            next_token_logits = outputs.logits[:, -1, :]
            
     
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
     
                sorted_indices_to_remove = cumulative_probs > top_p
      
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
    
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id)
            
    
            if unfinished_sequences.max() == 0:
                break
                
        return generated_tokens

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.token_embedding = new_embeddings

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
    ) -> torch.Tensor:
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
            
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _prepare_decoder_attention_mask(
        self, attention_mask: Optional[torch.Tensor], input_shape: Tuple[int, int], inputs_embeds: torch.Tensor, past_key_values_length: int
    ) -> Optional[torch.Tensor]:

        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        causal_mask = self._make_causal_mask(
            input_shape,
            dtype,
            device=device,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            return causal_mask

        if attention_mask.dim() == 2:

            expanded_mask = attention_mask[:, None, None, :].expand(
                input_shape[0], 1, input_shape[1], input_shape[1] + past_key_values_length
            ).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            padding_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
            return padding_mask + causal_mask

        return causal_mask

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
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_ids.shape, x, past_key_values_length
        )
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv, use_cache, attention_mask)
            if use_cache and present_kv is not None:
                present_kvs.append(present_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (logits,) + (tuple(present_kvs) if present_kvs else None,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(present_kvs) if present_kvs else None,
            hidden_states=None,
            attentions=None,
        )