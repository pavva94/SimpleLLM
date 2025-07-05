import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """Individual expert in MoE layer"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        return self.ffn(x)


class TopKGate(nn.Module):
    """Top-K gating mechanism for MoE"""

    def __init__(self, dim, num_experts, top_k=2, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Gating network
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Add noise for load balancing during training
        self.noise_std = 1e-2

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch_size * seq_len, dim)

        # Compute gate logits
        gate_logits = self.gate(x_flat)

        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # Compute gating probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Compute load balancing loss
        gate_probs = F.softmax(gate_logits, dim=-1)
        load_balancing_loss = self._compute_load_balancing_loss(gate_probs)

        return top_k_indices, top_k_probs, load_balancing_loss

    def _compute_load_balancing_loss(self, gate_probs):
        """Compute load balancing loss to encourage equal expert usage"""
        # Mean probability of each expert being selected
        mean_probs = gate_probs.mean(dim=0)

        # Coefficient of variation (encourages uniform distribution)
        cv_loss = (mean_probs.std() / (mean_probs.mean() + 1e-10)) ** 2

        return cv_loss


class MoELayer(nn.Module):
    """Mixture of Experts Layer"""

    def __init__(self, dim, num_experts=8, top_k=2, hidden_dim=None, capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # Create experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])

        # Gating mechanism
        self.gate = TopKGate(dim, num_experts, top_k, capacity_factor)

        # Load balancing coefficient
        self.load_balancing_coef = 0.01

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch_size * seq_len, dim)

        # Get gating decisions
        top_k_indices, top_k_probs, load_balancing_loss = self.gate(x_flat)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)

            if expert_mask.sum() > 0:
                # Get tokens for this expert
                expert_tokens = x_flat[expert_mask]

                # Process through expert
                expert_output = expert(expert_tokens)

                # Get the gating weights for this expert
                expert_indices = (top_k_indices == i).nonzero(as_tuple=True)
                expert_weights = top_k_probs[expert_indices]

                # Aggregate expert outputs weighted by gating probabilities
                for j, token_idx in enumerate(expert_indices[0]):
                    if token_idx < len(expert_output):
                        output[token_idx] += expert_weights[j] * expert_output[j]

        # Reshape back to original shape
        output = output.view(batch_size, seq_len, dim)

        # Add load balancing loss
        self.load_balancing_loss = self.load_balancing_coef * load_balancing_loss

        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE"""

    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with optional MoE"""

    def __init__(self, dim, n_heads, max_seq_len=2048, use_moe=False, num_experts=8, top_k=2):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.use_moe = use_moe

        if use_moe:
            self.feed_forward = MoELayer(dim, num_experts, top_k)
        else:
            self.feed_forward = SwiGLU(dim)

    def forward(self, x, mask=None):
        # Pre-norm with residual connection
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ModernMoELLM(nn.Module):
    """Modern LLM with Mixture of Experts"""

    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8, max_seq_len=2048,
                 moe_layers=None, num_experts=8, top_k=2):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.moe_layers = moe_layers or []  # List of layer indices to use MoE

        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Create transformer layers with optional MoE
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim, n_heads, max_seq_len,
                use_moe=(i in self.moe_layers),
                num_experts=num_experts,
                top_k=top_k
            ) for i in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # For tracking MoE losses
        self.moe_loss_coef = 0.01

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        seq_len = input_ids.size(1)

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Apply transformer blocks and collect MoE losses
        total_moe_loss = 0
        for layer in self.layers:
            x = layer(x, mask)

            # Collect MoE load balancing losses
            if hasattr(layer.feed_forward, 'load_balancing_loss'):
                total_moe_loss += layer.feed_forward.load_balancing_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Main cross-entropy loss
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Total loss includes MoE load balancing
            loss = ce_loss + self.moe_loss_coef * total_moe_loss

        return {
            "loss": loss,
            "logits": logits,
            "moe_loss": total_moe_loss if hasattr(self, 'moe_loss_coef') else 0
        }

    def get_expert_usage_stats(self):
        """Get statistics about expert usage"""
        stats = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer.feed_forward, 'gate'):
                stats[f'layer_{i}'] = {
                    'num_experts': layer.feed_forward.num_experts,
                    'top_k': layer.feed_forward.top_k
                }
        return stats


class TextDataset(Dataset):
    """Dataset for text training"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and truncate
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze()
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()  # For next token prediction
        }


def train_moe_model():
    # Configuration
    config = {
        "vocab_size": 50257,  # GPT-2 vocab size
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "max_seq_len": 512,
        "moe_layers": [2, 4, 6],  # Which layers to use MoE
        "num_experts": 8,
        "top_k": 2,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "num_epochs": 3,
        "warmup_steps": 1000,
        "max_steps": 10000,
        "moe_loss_coef": 0.01,
    }

    # Initialize wandb for logging
    wandb.init(project="moe-llm-training", config=config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"]) > 50]
    texts = texts[:10000]  # Use subset for demo

    # Create dataset and dataloader
    train_dataset = TextDataset(texts, tokenizer, config["max_seq_len"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize MoE model
    model = ModernMoELLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"],
        moe_layers=config["moe_layers"],
        num_experts=config["num_experts"],
        top_k=config["top_k"]
    ).to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Print MoE configuration
    print(f"MoE layers: {config['moe_layers']}")
    print(f"Experts per MoE layer: {config['num_experts']}")
    print(f"Top-K routing: {config['top_k']}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    def lr_lambda(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * step / config["max_steps"]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    step = 0

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        epoch_moe_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        for batch in progress_bar:
            if step >= config["max_steps"]:
                break

            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            moe_loss = outputs.get("moe_loss", 0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            epoch_moe_loss += moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss
            step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "moe_loss": f"{moe_loss:.4f}" if isinstance(moe_loss, torch.Tensor) else f"{moe_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "moe_loss": moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "step": step
            })

            # Save checkpoint periodically
            if step % 1000 == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "config": config
                }
                torch.save(checkpoint, f"moe_checkpoint_step_{step}.pt")

        avg_loss = epoch_loss / len(train_loader)
        avg_moe_loss = epoch_moe_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, MoE Loss: {avg_moe_loss:.4f}")

        if step >= config["max_steps"]:
            break

    # Save final model
    torch.save(model.state_dict(), "modern_moe_llm_final.pt")

    # Print expert usage statistics
    print("\nExpert usage statistics:")
    stats = model.get_expert_usage_stats()
    for layer_name, layer_stats in stats.items():
        print(f"{layer_name}: {layer_stats}")

    wandb.finish()
    print("Training completed!")
    return model


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text using the trained MoE model"""
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs["logits"]

            # Apply temperature
            next_token_logits = logits[0, -1, :] / temperature

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            # Stop at EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Train the MoE model
    model = train_moe_model()

    # Example text generation
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The future of artificial intelligence"
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nGenerated text:\n{generated}")