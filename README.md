# Modern LLM Training with GPU Support

A PyTorch implementation of a modern Large Language Model with state-of-the-art architectural improvements, designed for GPU training.

## Features

- **Modern Architecture**: RMSNorm, SwiGLU activation, Rotary Position Embeddings (RoPE)
- **GPU Accelerated**: Full CUDA support with optimized training
- **Public Dataset**: Trains on WikiText-103 dataset
- **Advanced Training**: AdamW optimizer, cosine scheduling, gradient clipping
- **Experiment Tracking**: Weights & Biases integration
- **Text Generation**: Built-in inference capabilities

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets wandb tqdm
```

### Basic Usage

1. **Clone and run:**
```bash
python train_llm.py
```

2. **Monitor training** (optional):
```bash
wandb login  # Setup W&B account first
```

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 50,257 | GPT-2 tokenizer vocabulary |
| `dim` | 512 | Model hidden dimension |
| `n_layers` | 8 | Number of transformer blocks |
| `n_heads` | 8 | Number of attention heads |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 5e-4 | Peak learning rate |
| `max_steps` | 10,000 | Total training steps |

## Architecture Details

### ModernLLM Class Structure

The `ModernLLM` class implements a state-of-the-art transformer architecture with several key improvements over traditional models:

```python
class ModernLLM(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8, max_seq_len=2048):
        # Token embeddings (no positional embeddings - using RoPE instead)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        
        # Final normalization and output projection
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
```

### Core Architectural Innovations

#### 1. **RMSNorm (Root Mean Square Normalization)**
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```
- **Why it's better**: More stable training than LayerNorm, fewer parameters
- **Key difference**: Only normalizes by RMS, no mean centering
- **Used in**: LLaMA, PaLM, and other modern LLMs

#### 2. **Rotary Position Embeddings (RoPE)**
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
- **Advantage**: Relative position encoding that generalizes to longer sequences
- **Key benefit**: Better extrapolation to sequences longer than training length
- **How it works**: Rotates query/key vectors by position-dependent angles

#### 3. **SwiGLU Activation Function**
```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SiLU(x*W1) * (x*W3) * W2
```
- **Formula**: `SiLU(xW₁) ⊙ (xW₃)W₂` where `⊙` is element-wise multiplication
- **Benefits**: Better performance than ReLU/GELU, used in PaLM and LLaMA
- **Architecture**: Uses 3 linear layers instead of 2 (gate mechanism)

#### 4. **Pre-Normalization Architecture**
```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        # Pre-norm: normalize BEFORE attention, not after
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
```
- **Key change**: Normalization happens before attention/FFN, not after
- **Benefit**: More stable training, better gradient flow
- **Used in**: GPT-2, LLaMA, and most modern transformers

### Detailed Component Breakdown

#### MultiHeadAttention with RoPE
```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, mask=None):
        # 1. Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 2. Apply rotary embeddings to Q and K
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 3. Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)
        
        # 4. Apply causal mask and softmax
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # 6. Concatenate heads and project
        return self.o_proj(out.reshape(batch_size, seq_len, self.dim))
```

#### Feed-Forward Network (SwiGLU)
```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        hidden_dim = hidden_dim or int(dim * 8/3)  # ~2.67x expansion (vs 4x in standard FFN)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # Output projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # Value projection
        
    def forward(self, x):
        # Gate mechanism: SiLU(xW1) acts as gate for xW3
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

#### Model Update with Mixture of Experts
##### Mixture of Experts (MoE) vs SwiGLU

##### Why MoE Over Standard SwiGLU?

Mixture of Experts (MoE) represents a paradigm shift from the traditional "one-size-fits-all" approach of standard feed-forward networks like SwiGLU. While SwiGLU provides excellent activation dynamics through its gated mechanism, it processes every token through the same parameters, creating a computational bottleneck as models scale. MoE breaks this limitation by creating specialized "expert" networks that can focus on different aspects of the input - some experts might specialize in mathematical reasoning, others in creative writing, and others in factual recall. The key insight is that for any given token, you typically only need a small subset of the model's total capacity. By routing each token to only the top-k most relevant experts (usually k=2), MoE achieves dramatically increased model capacity while maintaining similar computational costs. This sparsity is crucial for scaling: instead of making every parameter work on every token, MoE allows the model to grow its capacity by adding more experts without proportionally increasing inference cost. The routing mechanism learns to make intelligent decisions about which experts to activate, creating an implicit form of conditional computation that adapts to the input's complexity and type.

##### Implementation Comparison

##### Traditional SwiGLU (Dense)
```python
class SwiGLU(nn.Module):
    """Standard SwiGLU - ALL tokens through ALL parameters"""
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # Up
        
    def forward(self, x):
        # Every token uses the same computation path
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

##### MoE Layer (Sparse)
```python
class MoELayer(nn.Module):
    """Mixture of Experts - tokens routed to specialized experts"""
    
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            SwiGLU(dim) for _ in range(num_experts)
        ])
        
        # Routing mechanism - learns which expert to use
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        # Compute routing decisions
        gate_logits = self.gate(x_flat)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Route tokens to selected experts
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Only process tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.sum() > 0:
                expert_tokens = x_flat[expert_mask]
                expert_output = expert(expert_tokens)
                
                # Weighted combination based on gating probabilities
                expert_indices = (top_k_indices == i).nonzero(as_tuple=True)
                expert_weights = top_k_probs[expert_indices]
                
                for j, token_idx in enumerate(expert_indices[0]):
                    output[token_idx] += expert_weights[j] * expert_output[j]
        
        return output.view(batch_size, seq_len, dim)
```

#### Key Advantages

- **🚀 Massive Capacity**: 8 experts with top-2 routing = 4x parameters, same compute cost
- **🎯 Specialization**: Each expert can focus on specific patterns or domains
- **📈 Scalability**: Add more experts without proportional compute increase
- **🧠 Adaptive**: Routing learns to match token complexity with expert capacity
- **⚡ Efficiency**: Only 25% of parameters active per token (2 out of 8 experts)

##### Usage in Model

```python
# Mix MoE and regular layers for optimal performance
model = ModernMoELLM(
    vocab_size=50257,
    dim=512,
    n_layers=8,
    moe_layers=[2, 4, 6],  # Use MoE in specific layers
    num_experts=8,
    top_k=2
)
```

This sparse activation pattern allows the model to scale to billions of parameters while maintaining reasonable inference costs, making it ideal for large language models that need to handle diverse tasks and domains.

### Model Initialization Strategy

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # Xavier/Glorot normal initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

### Key Design Decisions

1. **No Bias Terms**: Modern LLMs often omit bias terms for better scaling
2. **Tied Embeddings**: Input and output embeddings could be tied (not implemented here)
3. **Causal Masking**: Ensures autoregressive generation (can't see future tokens)
4. **Weight Sharing**: The `lm_head` could share weights with token embeddings

### Performance Characteristics

| Component | Parameters | Computation | Memory Impact |
|-----------|------------|-------------|---------------|
| Token Embedding | `vocab_size × dim` | O(1) per token | Linear in vocab size |
| Attention Layers | `4 × dim²` per layer | O(seq_len²) | Quadratic in sequence length |
| SwiGLU FFN | `3 × dim × hidden_dim` | O(seq_len × dim²) | Linear in sequence length |
| RMSNorm | `dim` per layer | O(seq_len × dim) | Minimal |

### Model Size Scaling

For the default configuration:
- **Embedding**: 50,257 × 512 = 25.7M parameters
- **8 Attention Layers**: 8 × (4 × 512²) = 8.4M parameters  
- **8 SwiGLU Layers**: 8 × (3 × 512 × 1,365) = 16.8M parameters
- **Total**: ~25M parameters

### Memory Usage Breakdown

During training with batch_size=16, seq_len=512:
- **Model Parameters**: ~100MB (25M × 4 bytes)
- **Gradients**: ~100MB (same as parameters)
- **Optimizer States**: ~200MB (AdamW stores momentum + variance)
- **Activations**: ~500MB (depends on batch size and sequence length)
- **Total**: ~1GB GPU memory minimum

## Training Process

1. **Dataset Loading**: Downloads WikiText-103 automatically
2. **Preprocessing**: Tokenizes text with GPT-2 tokenizer
3. **Training Loop**: 
   - Causal language modeling objective
   - Gradient accumulation and clipping
   - Learning rate warmup and cosine decay
4. **Checkpointing**: Saves model every 1,000 steps
5. **Text Generation**: Test generation after training

## Customization

### Adjust Model Size

```python
config = {
    "dim": 768,        # Larger hidden dimension
    "n_layers": 12,    # More transformer blocks
    "n_heads": 12,     # More attention heads
    # ... other parameters
}
```

### Use Different Dataset

```python
# Replace WikiText with your preferred dataset
dataset = load_dataset("your_dataset_name", split="train")
```

### Modify Training Settings

```python
config = {
    "batch_size": 32,      # Larger batches (needs more GPU memory)
    "learning_rate": 1e-4, # Lower learning rate
    "max_steps": 50000,    # Longer training
}
```

## Output Files

- `checkpoint_step_*.pt`: Training checkpoints
- `modern_llm_final.pt`: Final trained model
- `wandb/`: Experiment logs (if using W&B)

## Text Generation

After training, the model automatically generates sample text:

```python
prompt = "The future of artificial intelligence"
generated = generate_text(model, tokenizer, prompt)
print(generated)
```

## System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ free space for dataset and checkpoints
- **Python**: 3.8+ with PyTorch 1.12+

## Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `batch_size` in config
- Reduce `max_seq_len` or `dim`

**Slow Training:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `num_workers` in DataLoader

**Dataset Download Issues:**
- Check internet connection
- Try different dataset: `load_dataset("wikitext", "wikitext-2-raw-v1")`

## Performance Tips

- Use mixed precision training for faster training:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

- Enable compilation for PyTorch 2.0+:
```python
model = torch.compile(model)
```

- Monitor GPU usage:
```bash
nvidia-smi
```

## License

MIT License - Feel free to use and modify for your projects.