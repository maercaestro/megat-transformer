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



# Load configuration
config = load_config()

# Ensure checkpoint directory exists
checkpoint_dir = config["training"]["checkpoint_dir"]
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="transformer_training", config=config)

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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the Transformer model
model = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    h=N_HEADS,
    d_ff=D_FF,
    src_vocab_size=len(source_vocab.vocab),
    tgt_vocab_size=len(target_vocab.vocab),
    max_len=MAX_LEN,
    dropout=config["model"].get("dropout_rate", 0.1)
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Helper function to create a square mask for the target sequence
def create_tgt_mask(tgt_seq_len):
    mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).bool()
    return mask[:tgt_seq_len-1, :tgt_seq_len-1]  # Align with tgt sequence excluding last token

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        source_seq = batch['source_seq']
        target_seq = batch['target_seq']
        
        # Create a square target mask of shape matching target_seq[:, :-1]
        tgt_seq_len = target_seq.size(1)
        tgt_mask = create_tgt_mask(tgt_seq_len).to(target_seq.device)

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

    # Save model checkpoint
    checkpoint_path = f"{config['training']['checkpoint_dir']}transformer_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path)

# Finish wandb logging
wandb.finish()
