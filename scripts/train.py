import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset.custom_dataset import CustomDataset, BuildVocabulary
from src.transformer import Transformer
import pandas as pd
import wandb
from config.config import load_config
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
config = load_config()

# Initialize wandb
wandb.init(project="transformer_training", config=config, resume="allow")

# Hyperparameters from config
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
EPOCHS = config["training"]["epochs"]
NUM_LAYERS = config["model"]["num_layers"]
D_MODEL = config["model"]["d_model"]
N_HEADS = config["model"]["num_heads"]
D_FF = config["model"]["d_ff"]
SOURCE_MAX_LEN = config["data"]["source_max_len"]
TARGET_MAX_LEN = config["data"]["target_max_len"]
MAX_LEN = max(SOURCE_MAX_LEN, TARGET_MAX_LEN)

# Load data and create vocabulary
dataset_path = config["data"]["dataset_path"]
df = pd.read_csv(dataset_path)

# Build vocabularies for source and target
source_vocab = BuildVocabulary(df['source_text'].tolist())
target_vocab = BuildVocabulary(df['translated_text'].tolist())

# Initialize dataset and dataloader
train_dataset = CustomDataset(df, source_vocab, target_vocab, SOURCE_MAX_LEN, TARGET_MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Initialize the Transformer model and move it to the GPU
model = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    h=N_HEADS,
    d_ff=D_FF,
    src_vocab_size=len(source_vocab.vocab),
    tgt_vocab_size=len(target_vocab.vocab),
    max_len=MAX_LEN,
    dropout=config["model"].get("dropout_rate", 0.1)
).to(device)  # Move model to GPU/CPU based on availability

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.vocab["<pad>"]).to(device)  # Move loss function to device
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Check for manually uploaded checkpoint
checkpoint_path = "/content/transformer_epoch_10.pth"
start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    except KeyError:
        model.load_state_dict(checkpoint)
        print("Loaded model weights only. Starting from epoch 0 without optimizer state.")

# Set checkpoint path to root directory
latest_checkpoint = "/content/latest_checkpoint.pth"

# Helper function to create a square mask for the target sequence
def create_tgt_mask(tgt_seq_len, device):
    mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=device)).bool()
    return mask[:tgt_seq_len-1, :tgt_seq_len-1]  # Align with tgt sequence excluding last token

# Training loop
try:
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            source_seq = batch['source_seq'].to(device)  # Move source sequence to GPU
            target_seq = batch['target_seq'].to(device)  # Move target sequence to GPU

            # Create a square target mask of shape matching target_seq[:, :-1]
            tgt_seq_len = target_seq.size(1)
            tgt_mask = create_tgt_mask(tgt_seq_len, device)

            # Forward pass
            optimizer.zero_grad()
            output = model(source_seq, target_seq[:, :-1], tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq[:, 1:].reshape(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Log the loss to wandb
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        # Save model checkpoint after each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, latest_checkpoint)
        print(f"Checkpoint saved at {latest_checkpoint}")
        wandb.save(latest_checkpoint)

except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, latest_checkpoint)
    wandb.save(latest_checkpoint)
    print("Checkpoint saved and uploaded to wandb.")
    wandb.finish()

wandb.finish()
