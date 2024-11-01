import sys
import os
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
from src.transformer import Transformer
from config.config import load_config
from dataset.custom_dataset import CustomDataset  

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs, targets = batch['input'].to(device), batch['target'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Load configuration from YAML file
    config = load_config()
    
    # Initialize Weights & Biases (W&B)
    wandb.init(
        project="megat-transformer",
        config=config,
        name="Transformer Training on 300,000 Rows",
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with parameters from config.yaml
    model = Transformer(
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        h=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Initialize vocab and dataset
    vocab = config['data']['vocab']  # Assume vocab is loaded from config.yaml
    max_len = config['model']['max_len']
    dataset = CustomDataset(data_path=config['data']['data_path'], vocab=vocab, max_len=max_len)
    
    # Select 300,000 rows and split into training (80%) and validation (20%)
    subset_indices = list(range(300000))  # Use only the first 300,000 rows
    train_indices, val_indices = train_test_split(subset_indices, test_size=0.2, random_state=42)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_dataloader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        
        # Log training loss to W&B
        wandb.log({"epoch": epoch+1, "train_loss": train_loss})
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {train_loss:.4f}")
        
        # Optionally save the model after each epoch to W&B and local storage
        if config['training']['save_model']:
            model_save_path = f"{config['training']['model_save_path']}model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)  # Save checkpoint to W&B for future access

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
