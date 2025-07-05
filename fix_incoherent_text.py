from main import ModernLLM, TextDataset
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm




def generate_coherent_text(model, tokenizer, prompt, max_length=50, temperature=0.3, top_k=20, repetition_penalty=1.1):
    """Generate more coherent text with better sampling"""
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = input_ids.clone()

    with torch.no_grad():
        for step in range(max_length):
            outputs = model(input_ids)
            logits = outputs["logits"][0, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens[0].tolist()):
                    logits[token_id] /= repetition_penalty

            # Lower temperature for more focused generation
            logits = logits / temperature

            # Top-k sampling with smaller k
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, -float('inf'))
                logits[top_k_indices] = top_k_logits

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Stop at sentence endings or repetition
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Add to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

            # Stop if we hit a period followed by space (end of sentence)
            if len(generated_tokens[0]) > 2:
                last_two = tokenizer.decode(generated_tokens[0, -2:])
                if '. ' in last_two or '.\n' in last_two:
                    break

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


# Test with better settings
def test_coherent_generation():
    # Load your trained model
    model = ModernLLM(vocab_size=50257, dim=512, n_layers=8, n_heads=8)
    model.load_state_dict(torch.load("modern_llm_final.pt"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "The future of artificial intelligence will",
        "Scientists have discovered that",
        "In the year 2030, technology will",
        "The most important thing about",
        "Once upon a time there was"
    ]

    print("=== COHERENT GENERATION TEST ===")
    for prompt in prompts:
        generated = generate_coherent_text(
            model, tokenizer, prompt,
            max_length=30,  # Shorter for better coherence
            temperature=0.3,  # Lower temperature
            top_k=20,  # Smaller top_k
            repetition_penalty=1.1
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)


# BETTER TRAINING DATA FILTERING
def get_clean_dataset():
    """Get cleaner training data"""
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    clean_texts = []
    for item in dataset:
        text = item["text"].strip()

        # Filter out bad samples
        if len(text) < 100:  # Too short
            continue
        if text.startswith("="):  # Wikipedia headers
            continue
        if "(" in text and ")" in text and text.count("(") > 3:  # Too many parentheses
            continue
        if any(char in text for char in ["µ", "â", "ê", "ë", "î"]):  # Encoding issues
            continue
        if text.count(".") < 2:  # Not enough sentences
            continue

        # Only keep texts that start with capital letter
        if text[0].isupper():
            clean_texts.append(text)

        if len(clean_texts) >= 50000:  # Enough clean data
            break

    print(f"Filtered to {len(clean_texts)} clean texts")
    return clean_texts


# RESTART TRAINING WITH CLEAN DATA
def retrain_with_clean_data():
    """Restart training with better data"""
    config = {
        "vocab_size": 50257,
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "max_seq_len": 256,  # Shorter sequences for better coherence
        "batch_size": 32,  # Larger batch
        "learning_rate": 3e-4,  # Slightly higher LR
        "num_epochs": 5,
        "max_steps": 20000,  # More training
        "warmup_steps": 1000,
    }

    # Initialize wandb for logging
    wandb.init(project="modern-llm-training", config=config)

    # Get clean data
    clean_texts = get_clean_dataset()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset with clean data
    dataset = TextDataset(clean_texts, tokenizer, config["max_seq_len"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize fresh model
    model = ModernLLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["max_seq_len"]
    )

    model.to(device)

    print("Starting training with clean data...")
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


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If still bad, retrain with clean data
    retrain_with_clean_data()

    # Then, test your current model with better generation
    test_coherent_generation()