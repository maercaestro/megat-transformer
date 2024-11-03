import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset.custom_dataset import CustomDataset, BuildVocabulary
from src.transformer import Transformer
import pandas as pd
import wandb
from config import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize wandb for evaluation
wandb.init(project="transformer_evaluation", config=config)

# Hyperparameters from config
BATCH_SIZE = config["training"]["batch_size"]
NUM_LAYERS = config["model"]["num_layers"]
D_MODEL = config["model"]["d_model"]
N_HEADS = config["model"]["num_heads"]
D_FF = config["model"]["d_ff"]
SOURCE_MAX_LEN = config["data"]["source_max_len"]
TARGET_MAX_LEN = config["data"]["target_max_len"]

# Load data and create vocabulary
dataset_path = config["data"]["dataset_path"]
df = pd.read_csv(dataset_path)

# Build vocabularies for source and target
source_vocab = BuildVocabulary(df['source_text'].tolist())
target_vocab = BuildVocabulary(df['translated_text'].tolist())

# Initialize dataset and dataloader for evaluation
eval_dataset = CustomDataset(df, source_vocab, target_vocab, SOURCE_MAX_LEN, TARGET_MAX_LEN)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the Transformer model
model = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    vocab_size=len(source_vocab.vocab),
    num_heads=N_HEADS,
    d_ff=D_FF
)

# Load the latest checkpoint
checkpoint_path = f"{config['training']['checkpoint_dir']}transformer_epoch_{config['training']['epochs']}.pth"
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Define loss function for evaluation
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.vocab["<pad>"])

# Evaluation loop
total_loss = 0
with torch.no_grad():
    for batch in eval_loader:
        source_seq = batch['source_seq']
        target_seq = batch['target_seq']
        source_mask = batch['source_mask']
        target_mask = batch['target_mask']

        # Forward pass
        output = model(source_seq, target_seq[:, :-1], source_mask, target_mask[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), target_seq[:, 1:].reshape(-1))
        
        total_loss += loss.item()

# Calculate and log average loss
avg_loss = total_loss / len(eval_loader)
print(f"Evaluation Loss: {avg_loss:.4f}")

# Log evaluation loss to wandb
wandb.log({"evaluation_loss": avg_loss})

# Finish wandb logging
wandb.finish()
