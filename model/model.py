import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass


torch.manual_seed(1024)

@dataclass
class EtudeConfig:
    batch_size: int = 4
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 512
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 20000 
    eos_token_id: int = 0   
    use_moe: bool = False
    expert_number: int = 2
    top_k: int = 1
    shared_experts_number: int = 1


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=torch.device("cpu"), dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.hidden_size = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.rope = RotaryEmbedding(self.head_size)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_size)  
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  

        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    
        cos, sin = self.rope(q, seq_len=T) 
        
  
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out



class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = SwiGLU(
            dim=config.n_embd,
            hidden_dim=4 * config.n_embd,
            dropout=config.dropout
        )

    def forward(self, x):
        return self.net(x)


class BasicExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = SwiGLU(
            dim=dim,
            hidden_dim=4 * dim 
        )

    def forward(self, x):
        return self.net(x)

class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k, noisy_gate=True):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
        self.noisy_gate = noisy_gate

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)

        if self.noisy_gate:
            noise = torch.randn_like(router_logits) * 1e-2
            router_logits = router_logits + noise

        routing_probs = F.softmax(router_logits, dim=-1)
        router_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        return router_logits, router_weights, selected_experts

class MOEConfig:
    def __init__(self, hidden_dim, expert_number, top_k, shared_experts_number=2):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList([BasicExpert(self.hidden_dim) for _ in range(self.expert_number)])

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k, noisy_gate=True)

    def forward(self, x, mask=None):
        B, T, H = x.size()
        hidden_states = x.view(-1, H)  # [B*T, H]

        if mask is not None:
            mask_flat = mask.view(-1)

            hidden_states_masked = hidden_states[mask_flat]
        else:
            hidden_states_masked = hidden_states
            mask_flat = None

        router_logits, router_weights, selected_experts = self.router(hidden_states_masked)

        num_tokens, top_k = selected_experts.shape
        
        flat_top_k_indices = selected_experts.view(-1) 
        flat_hidden_states = hidden_states_masked.repeat_interleave(top_k, dim=0) 
        flat_router_weights = router_weights.view(-1, 1) 

        expert_capacity = num_tokens * top_k // self.expert_number 
        expert_bins = torch.bincount(flat_top_k_indices, minlength=self.expert_number)
        expert_cumsum = torch.cumsum(expert_bins, dim=0)
        
        sorted_indices = torch.argsort(flat_top_k_indices)

        permuted_hidden_states = flat_hidden_states[sorted_indices]
        permuted_weights = flat_router_weights[sorted_indices]

        split_hidden_states = torch.split(permuted_hidden_states, expert_bins.tolist(), dim=0)

        results = []
        for i, expert in enumerate(self.experts):
            if split_hidden_states[i].shape[0] > 0: 
                results.append(expert(split_hidden_states[i]))


        permuted_output = torch.cat(results, dim=0)
        

        permuted_output = permuted_output * permuted_weights
        

        unpermuted_output = torch.zeros_like(permuted_output)
        unpermuted_output[sorted_indices] = permuted_output
        token_outputs = unpermuted_output.view(num_tokens, top_k, H).sum(dim=1)
        final_hidden = torch.zeros_like(hidden_states)
        if mask is not None:
            final_hidden[mask_flat] = token_outputs
        else:
            final_hidden = token_outputs

        final_hidden = final_hidden.view(B, T, H)
        return final_hidden, router_logits, selected_experts
    
class Block(nn.Module):
    def __init__(self, config: EtudeConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)

        self.ln1 = RMSNorm(config.n_embd)
        self.use_moe = config.use_moe
        if self.use_moe:
            moe_config = MOEConfig(
                hidden_dim=config.n_embd,
                expert_number=config.expert_number,
                top_k=config.top_k
            )
            self.ffn = SparseMOE(moe_config)
        else:

            self.ffn = FeedForward(config)

        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x):
        x_att = self.att(self.ln1(x))
        x = x + x_att
        if self.use_moe:
            x_ffn, router_logits, selected_experts = self.ffn(self.ln2(x))
            x = x + x_ffn

            return x, router_logits, selected_experts
        else:
            x_ffn = self.ffn(self.ln2(x))
            x = x + x_ffn

            return x, None, None


class Etude(nn.Module):
    def __init__(self, config: EtudeConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
            padding_idx=None
        )
        self.n_embd = config.n_embd

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_head.weight = self.token_embedding.weight

        self.eos_token = config.eos_token_id 


    def forward(self, idx, targets=None):
        B, T = idx.shape

        mask = (idx != self.eos_token)

        x = self.token_embedding(idx) 
        total_aux_loss = 0.0
        aux_coef = 0.01

        for block in self.blocks:
            x, router_logits, selected_experts = block(x)
            if router_logits is not None and selected_experts is not None:

                router_logits_flat = router_logits.view(-1, router_logits.size(-1))
                selected_experts_flat = selected_experts.view(-1, selected_experts.size(-1))
                mask_flat = mask.view(-1)
                router_logits_masked = router_logits_flat[mask_flat]
                selected_experts_masked = selected_experts_flat[mask_flat]
                if router_logits_masked.numel() > 0:
                    aux_loss = self.compute_aux_loss(router_logits_masked, selected_experts_masked)
                    total_aux_loss += aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if targets is not None:

            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.eos_token
            )
            loss = ce_loss + aux_coef * total_aux_loss

        return logits, loss


    def compute_aux_loss(self, router_logits, selected_experts):

        num_experts = router_logits.size(-1)
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]


        expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
        expert_mask = expert_mask.sum(dim=1).float()  # [N, E]


        load = router_probs.mean(dim=0)  # [E]
        usage = expert_mask.mean(dim=0)  # [E]
        aux_loss = (load * usage).sum() * num_experts

        return aux_loss


def connectivity_test_dynamic():
    print("--- Testing with SwiGLU FFN (MoE disabled by default) ---")
    config = EtudeConfig(use_moe=False) 
    model = Etude(config)

    train_seq_len = 16
    idx_train = torch.randint(0, config.vocab_size, (config.batch_size, train_seq_len))
    targets_train = torch.randint(0, config.vocab_size, (config.batch_size, train_seq_len))
    logits_train, loss_train = model(idx_train, targets_train)
    print("训练序列测试:")
    print("  输入 shape:", idx_train.shape)
    print("  logits shape:", logits_train.shape)
    print("  loss:", loss_train.item())
    import sys
    sys.stdout.flush()

    infer_seq_len = 30
    idx_infer = torch.randint(0, config.vocab_size, (config.batch_size, infer_seq_len))
    logits_infer, _ = model(idx_infer)
    print("推理序列测试:")
    print("  输入 shape:", idx_infer.shape)
    print("  logits shape:", logits_infer.shape)

    print("\n--- Testing with SwiGLU Experts (MoE enabled) ---")
    config_moe = EtudeConfig(use_moe=True) 
    model_moe = Etude(config_moe)

    logits_moe, loss_moe = model_moe(idx_train, targets_train)
    print("MoE训练序列测试:")
    print("  输入 shape:", idx_train.shape)
    print("  logits shape:", logits_moe.shape)
    print("  loss:", loss_moe.item())


if __name__ == "__main__":
    connectivity_test_dynamic()