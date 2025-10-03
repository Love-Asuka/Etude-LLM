# model.py
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
        
        # --- Gated Attention (全注意力) 配置 ---
        num_key_value_heads: Optional[int] = None,
        attention_bias: bool = False,
        rotary_pct: float = 0.5,

        # --- Gated DeltaNet (线性注意力) 配置 ---
        linear_num_value_heads: int = 4,
        linear_num_key_heads: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        hidden_act: str = "silu",

        # --- 混合注意力模式 ---
        layer_types: Optional[List[str]] = None,

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
        
        # 全注意力配置
        self.attention_bias = attention_bias
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else n_head
        self.rotary_pct = rotary_pct
        
        # 线性注意力配置
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.hidden_act = hidden_act

        # 派生属性
        self.head_size = self.n_embd // self.n_head

        self.use_cache = False
        
        # 混合注意力层类型
        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool(i % 2) else "full_attention"
                for i in range(self.n_layer)
            ]
        else:
            self.layer_types = layer_types
        
        if len(self.layer_types) != self.n_layer:
            raise ValueError("`layer_types` 列表的长度必须等于 `n_layer`")

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
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000, rotary_pct: float = 1.0, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rotary_pct = rotary_pct
        
        self.rotary_dim = int(self.dim * self.rotary_pct)
        self.rotary_dim -= self.rotary_dim % 2
        self.non_rotary_dim = self.dim - self.rotary_dim

        if self.rotary_dim > 0:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32) / self.rotary_dim))
            self.register_buffer("inv_freq", inv_freq)
        
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        if self.rotary_dim > 0:
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_rot = emb.cos()
            sin_rot = emb.sin()
        else:
            cos_rot = torch.empty(seq_len, 0, device=device)
            sin_rot = torch.empty(seq_len, 0, device=device)

        if self.non_rotary_dim > 0:
            cos_no_rot = torch.ones(seq_len, self.non_rotary_dim, device=device)
            sin_no_rot = torch.zeros(seq_len, self.non_rotary_dim, device=device)
        else:
            cos_no_rot = torch.empty(seq_len, 0, device=device)
            sin_no_rot = torch.empty(seq_len, 0, device=device)

        cos_cached = torch.cat((cos_rot, cos_no_rot), dim=-1)
        sin_cached = torch.cat((sin_rot, sin_no_rot), dim=-1)

        self.register_buffer("cos_cached", cos_cached.to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin_cached.to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2] # 注意：这里的 x 是 query，形状为 (B, n_head, T, head_size)
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

class GatedAttention(nn.Module):
    def __init__(self, config): # 假设 config 是一个包含所需参数的对象
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.n_embd = config.n_embd
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.n_head // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_size * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.n_embd, self.num_key_value_heads * self.head_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.n_embd, self.num_key_value_heads * self.head_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_head * self.head_size, self.n_embd, bias=config.attention_bias)
        
        self.q_norm = RMSNorm(self.head_size, eps=1e-6)
        self.k_norm = RMSNorm(self.head_size, eps=1e-6)
        
        self.rope = RotaryEmbedding(self.head_size, rotary_pct=config.rotary_pct, max_position_embeddings=4096)
        self.dropout = config.dropout
    
    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        
        q_gate, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q, gate = torch.chunk(q_gate.view(B, T, self.n_head, self.head_size * 2), 2, dim=-1)
        k = k.view(B, T, self.num_key_value_heads, self.head_size)
        v = v.view(B, T, self.num_key_value_heads, self.head_size)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2) # (B, n_head, T, head_size)
        k = k.transpose(1, 2) # (B, num_kv_heads, T, head_size)
        v = v.transpose(1, 2) # (B, num_kv_heads, T, head_size)

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
        
        k = self.repeat_kv(k, self.num_key_value_groups)
        v = self.repeat_kv(v, self.num_key_value_groups)

        is_causal = attention_mask is None
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = attn_output * F.sigmoid(gate.reshape(B, T, C))
        
        return self.o_proj(attn_output), present_key_value

class GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64, 
    initial_recurrent_state: Optional[torch.Tensor] = None
):
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    # 如果序列长度小于 chunk_size，则将 chunk_size 设为序列长度
    if sequence_length < chunk_size:
        chunk_size = sequence_length

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    if initial_recurrent_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    else:
        last_recurrent_state = initial_recurrent_state.to(value.device, value.dtype)

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

class GatedDeltaNet(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.act = F.silu if config.hidden_act == "silu" else F.gelu

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim,
            bias=False, kernel_size=self.conv_kernel_size,
            groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = GatedRMSNorm(self.head_v_dim, eps=1e-6)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        new_shape_qkvz = mixed_qkvz.size()[:-1] + (self.num_k_heads, 2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads)
        new_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)
        mixed_qkvz = mixed_qkvz.view(*new_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_shape_ba)
        split_qkvz = [self.head_k_dim, self.head_k_dim, (self.num_v_heads // self.num_k_heads * self.head_v_dim), (self.num_v_heads // self.num_k_heads * self.head_v_dim)]
        split_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_ba, dim=3)
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        
        # 1. 投影输入
        projected_qkvz = self.in_proj_qkvz(x)
        projected_ba = self.in_proj_ba(x)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_qkvz, projected_ba)
        query, key, value = (t.reshape(B, T, -1) for t in (query, key, value))
        
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # 2. 卷积和状态管理
        past_recurrent_state, past_conv_state = None, None
        if past_key_value is not None:
            past_recurrent_state, past_conv_state = past_key_value

        # Conv1d 需要 (B, C, T) 格式
        mixed_qkv = mixed_qkv.transpose(1, 2)
        
        if past_conv_state is not None:
            # 解码模式：拼接历史卷积状态和当前输入
            mixed_qkv = torch.cat([past_conv_state, mixed_qkv], dim=2)

        conv_out = self.act(self.conv1d(mixed_qkv))
        
        present_conv_state = None
        if use_cache:
            # 缓存最后 kernel_size - 1 个时间步用于下一次解码
            present_conv_state = conv_out[:, :, -(self.conv_kernel_size - 1):]
        
        # 截断卷积引入的额外 padding
        conv_out = conv_out[:, :, :T].transpose(1, 2)

        # 3. 分离 Q, K, V 并重塑
        query, key, value = torch.split(conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, T, -1, self.head_k_dim)
        key = key.reshape(B, T, -1, self.head_k_dim)
        value = value.reshape(B, T, -1, self.head_v_dim)

        # 4. 计算门控和衰减因子
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # 5. 执行 Gated Delta Rule
        core_attn_out, present_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_recurrent_state=past_recurrent_state
        )

        # 6. 后处理和输出投影
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(B, T, -1)

        output = self.out_proj(core_attn_out)
        
        present_key_value = (present_recurrent_state, present_conv_state) if use_cache else None
        
        return output, present_key_value

class FeedForward(nn.Module):
    def __init__(self, config: EtudeHFConfig):
        super().__init__()
        hidden_dim = int(2/3 * 4 * config.n_embd)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_exchanged = torch.empty_like(x)
        x_exchanged[:, 0, :] = x[:, 0, :]
        x_exchanged[:, 1:, :] = x[:, 1:, :] * 0.9 + x[:, :-1, :] * 0.1
        x = x_exchanged

        gate, up = self.w1(x).chunk(2, dim=-1)
        y = gate * torch.sigmoid(self.beta * gate + self.b) * up
        y = self.w2(y)
        return self.dropout(y)

class Block(nn.Module):
    def __init__(self, config: EtudeHFConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        if layer_type == "full_attention":
            self.att = GatedAttention(config)
        elif layer_type == "linear_attention":
            self.att = GatedDeltaNet(config)
        else:
            raise ValueError(f"未知的层类型: {layer_type}")
            
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
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
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
        
        past_key_values = None
        current_input_ids = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=current_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                    **kwargs
                )
            
            past_key_values = outputs.past_key_values
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
            current_input_ids = next_tokens.unsqueeze(-1) # 下一轮的输入只有新生成的 token
            
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
        
        past_key_values_length = 0
        if past_key_values is not None:
            for i, layer_type in enumerate(self.config.layer_types):
                if layer_type == "full_attention":
                    if past_key_values[i] is not None and isinstance(past_key_values[i], tuple) and len(past_key_values[i]) == 2:
                        past_key_values_length = past_key_values[i][0].shape[2]
                        break

        # GatedDeltaNet 层会接收到它，但其内部逻辑会忽略它
        prepared_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_ids.shape, x, past_key_values_length
        )
        
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv, use_cache, prepared_attention_mask)
            if use_cache and present_kv is not None:
                present_kvs.append(present_kv)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

if __name__ == "__main__":
    print("--- Testing Etude Model with Mixed Attention ---")

    batch_size = 2
    seq_len = 64
    
    # 创建一个混合注意力模型配置 (full, linear, full, linear, ...)
    config = EtudeHFConfig()
    print("\nModel Configuration (Default Alternating):")
    print(f"Layer types: {config.layer_types}")

    model = Etude(config)
    model.eval()
    print(f"\nModel instantiated successfully. Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\nInput shape (input_ids): {input_ids.shape}")

    # 执行前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    # 检查输出形状
    logits = outputs.logits
    expected_logits_shape = (batch_size, seq_len, config.vocab_size)
    print(f"\nOutput shape (logits): {logits.shape}")
    assert logits.shape == expected_logits_shape
    print("✅ Logits shape is correct.")

    # 检查 past_key_values (混合状态缓存)
    past_key_values = outputs.past_key_values
    assert len(past_key_values) == config.n_layer

    # --- 检查 Layer 0 (full_attention) 的缓存 ---
    print("\n--- Checking Layer 0 (full_attention) Cache ---")
    full_attn_cache = past_key_values[0]
    assert isinstance(full_attn_cache, tuple) and len(full_attn_cache) == 2
    key_cache, value_cache = full_attn_cache
    expected_kv_shape = (batch_size, config.num_key_value_heads, seq_len, config.head_size)
    print(f"Shape of Key cache: {key_cache.shape}")
    print(f"Expected Key cache shape: {expected_kv_shape}")
    assert key_cache.shape == expected_kv_shape
    print("✅ Full attention Key cache shape is correct.")

    # --- 检查 Layer 1 (linear_attention) 的缓存 ---
    print("\n--- Checking Layer 1 (linear_attention) Cache ---")
    linear_attn_cache = past_key_values[1]
    assert isinstance(linear_attn_cache, tuple) and len(linear_attn_cache) == 2
    recurrent_state, conv_state = linear_attn_cache
    
    expected_recurrent_state_shape = (batch_size, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)
    conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + config.linear_value_head_dim * config.linear_num_value_heads
    expected_conv_state_shape = (batch_size, conv_dim, config.linear_conv_kernel_dim - 1)
    
    print(f"Shape of Recurrent State: {recurrent_state.shape}")
    print(f"Expected Recurrent State shape: {expected_recurrent_state_shape}")
    assert recurrent_state.shape == expected_recurrent_state_shape
    print("✅ Linear attention Recurrent State shape is correct.")
    
    print(f"Shape of Conv State: {conv_state.shape}")
    print(f"Expected Conv State shape: {expected_conv_state_shape}")
    assert conv_state.shape == expected_conv_state_shape
    print("✅ Linear attention Conv State shape is correct.")

    # --- 测试增量解码 ---
    print("\n--- Testing Incremental Decoding (Generation) Step ---")
    # 取出第一次的缓存
    past = outputs.past_key_values
    # 准备下一次的输入 (B, 1)
    next_input_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
    
    with torch.no_grad():
        next_outputs = model(input_ids=next_input_ids, use_cache=True, past_key_values=past)
    
    # 检查新生成的 logits 和缓存
    next_logits = next_outputs.logits
    expected_next_logits_shape = (batch_size, 1, config.vocab_size)
    print(f"Shape of next logits: {next_logits.shape}")
    assert next_logits.shape == expected_next_logits_shape
    print("✅ Next logits shape is correct.")

    next_past = next_outputs.past_key_values
    # 检查 full_attention 缓存长度是否增加
    next_key_cache = next_past[0][0]
    expected_next_kv_len = seq_len + 1
    print(f"Length of next Key cache (full_attention): {next_key_cache.shape[2]}")
    assert next_key_cache.shape[2] == expected_next_kv_len
    print("✅ Full attention cache length is correctly updated.")

    # 检查 linear_attention 缓存形状是否保持不变
    next_recurrent_state, next_conv_state = next_past[1]
    print(f"Shape of next Recurrent State (linear_attention): {next_recurrent_state.shape}")
    assert next_recurrent_state.shape == expected_recurrent_state_shape
    print(f"Shape of next Conv State (linear_attention): {next_conv_state.shape}")
    assert next_conv_state.shape == expected_conv_state_shape
    print("✅ Linear attention cache shapes are correctly maintained.")

    print("\n--- Testing with reversed layer types to validate robust logic ---")
    # 创建一个第一层是 linear_attention 的配置
    reversed_layer_types = [
        "full_attention" if bool(i % 2) else "linear_attention"
        for i in range(config.n_layer)
    ]
    config_reversed = EtudeHFConfig(layer_types=reversed_layer_types)
    print(f"Reversed layer types: {config_reversed.layer_types}")
    model_reversed = Etude(config_reversed)
    model_reversed.eval()

    with torch.no_grad():
        outputs_rev = model_reversed(input_ids=input_ids, use_cache=True)
        past_rev = outputs_rev.past_key_values
        next_outputs_rev = model_reversed(input_ids=next_input_ids, use_cache=True, past_key_values=past_rev)

    print("✅ Model with linear attention as first layer passed forward and decoding step without error.")

    print("\n--- Test Completed Successfully! ---")
