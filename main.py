import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm


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
    """Modern Transformer block with RMSNorm and SwiGLU"""

    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.feed_forward = SwiGLU(dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        # Pre-norm with residual connection
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class ModernLLM(nn.Module):
    """Modern LLM with latest architectural improvements"""

    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

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

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}


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


def train_model():
    # Configuration
    config = {
        "vocab_size": 50257,  # GPT-2 vocab size
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "max_seq_len": 512,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "num_epochs": 3,
        "warmup_steps": 1000,
        "max_steps": 10000,
    }

    # Initialize wandb for logging
    wandb.init(project="modern-llm-training", config=config)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (using WikiText-103 as example)
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Filter and prepare texts
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

    # Initialize model
    model = ModernLLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"]
    ).to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup optimizer with weight decay
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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Logging
            epoch_loss += loss.item()
            step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
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
                torch.save(checkpoint, f"checkpoint_step_{step}.pt")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        if step >= config["max_steps"]:
            break

    # Save final model
    torch.save(model.state_dict(), "modern_llm_final.pt")
    wandb.finish()

    print("Training completed!")
    return model


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text using the trained model"""
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
    # Train the model
    model = train_model()

    # Example text generation
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "The future of artificial intelligence"
    generated = generate_text(model, tokenizer, prompt)
    print(f"\nGenerated text:\n{generated}")