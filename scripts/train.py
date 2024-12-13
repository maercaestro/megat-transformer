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
from torch.optim.lr_scheduler import ReduceLROnPlateau





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

# Split dataset into train and validation sets
train_df = df.sample(frac=0.8, random_state=42)  # 80% training
val_df = df.drop(train_df.index)  # 20% validation

# Build vocabularies for source and target
source_vocab = BuildVocabulary(df['source_text'].tolist())
target_vocab = BuildVocabulary(df['translated_text'].tolist())

# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_df, source_vocab, target_vocab, SOURCE_MAX_LEN, TARGET_MAX_LEN)
val_dataset = CustomDataset(val_df, source_vocab, target_vocab, SOURCE_MAX_LEN, TARGET_MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the Transformer model with dropout
model = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    h=N_HEADS,
    d_ff=D_FF,
    src_vocab_size=len(source_vocab.vocab),
    tgt_vocab_size=len(target_vocab.vocab),
    max_len=MAX_LEN,
    dropout=config["model"].get("dropout_rate",0.2)
).to(device)

# Watch the model with WANDB
wandb.watch(model, log="all")

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, scale_factor=2.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor  # New scaling factor
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # Avoid division by zero
        scale = self.scale_factor * (self.d_model ** -0.5)  # Amplified scale
        lr = scale * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [max(1e-5, base_lr * lr) for base_lr in self.base_lrs]  # Ensure a minimum LR



# Define loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.vocab["<pad>"]).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# Add warm-up scheduler
warmup_steps = config["training"].get("warmup_steps", 4000)  # Default warm-up steps
scheduler = WarmupScheduler(optimizer, d_model=D_MODEL, warmup_steps=warmup_steps)  


# Set checkpoint paths
checkpoint_path = "/root/checkpoints/latest_checkpoint.pth"
os.makedirs("/root/checkpoints", exist_ok=True)

# Load checkpoint if exists
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

# Helper function to create a square causal mask
def create_tgt_mask(tgt_seq_len, device):
    """
    Creates a square causal mask for the target sequence to prevent looking ahead.
    The mask must align with the input shape of the decoder.
    """
    mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, tgt_seq_len, tgt_seq_len]

# Helper function to calculate validation loss
def evaluate_loss(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            source_seq = batch['source_seq'].to(device)
            target_seq = batch['target_seq'].to(device)

            tgt_seq_len = target_seq.size(1) - 1  # Adjust for slicing target_seq[:, :-1]
            tgt_mask = create_tgt_mask(tgt_seq_len, device)

            output = model(source_seq, target_seq[:, :-1], tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop with early stopping but no learning rate adjustment
early_stopping_patience = 30  # Number of epochs with no improvement before stopping
epochs_no_improve = 0
best_val_loss = float('inf')

try:
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            source_seq = batch['source_seq'].to(device)
            target_seq = batch['target_seq'].to(device)

            tgt_seq_len = target_seq.size(1) - 1  # Adjust for slicing target_seq[:, :-1]
            tgt_mask = create_tgt_mask(tgt_seq_len, device)

            optimizer.zero_grad()

            # Forward pass
            output = model(source_seq, target_seq[:, :-1], tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq[:, 1:].reshape(-1))

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()  # Update learning rate

            total_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)

        # Calculate validation loss
        val_loss = evaluate_loss(model, val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log metrics to WANDB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0]
        })

        # Save model checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved.")

# Log the final model as a WANDB artifact
artifact = wandb.Artifact("transformer-model", type="model")
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)
print("Model uploaded to WANDB.")

wandb.finish()

