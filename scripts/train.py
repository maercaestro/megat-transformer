# train.py - Training script for Transformer model using YAML configuration and Weights & Biases (W&B) logging
import pandas as pd
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb  # For logging
from src.transformer import Transformer
from config.config import load_config  # YAML config loader
from dataset.custom_dataset import CustomDataset, VocabularyBuilder  # Import VocabularyBuilder

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        inputs, targets = batch['input'].to(device), batch['target'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs, targets)
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print loss every 10 iterations for progress tracking
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    # Load configuration from YAML file
    config = load_config()  # Ensure config.yaml is in the config folder
    
    # Initialize W&B for tracking
    wandb.init(
        project="megat-transformer",
        config=config,
        name="Transformer Training on 30,000 Rows",
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VocabularyBuilder and build vocabulary
    vocab_builder = VocabularyBuilder(min_freq=1)
    train_dataset_path = config['data']['dataset_path']
    all_texts = pd.concat(pd.read_csv(train_dataset_path)[["source_text", "translated_text"]].values.flatten())
    vocab = vocab_builder.build(all_texts)
    max_len = config['model']['max_len']  # Retrieve max_len from config, if specified

    # Initialize model with parameters from config.yaml and vocab size
    model = Transformer(
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        h=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        max_len=max_len,
        dropout=config['model']['dropout']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Load the dataset and create DataLoader
    train_dataset = CustomDataset(data_path=train_dataset_path, vocab=vocab, max_len=max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch)
        
        # Log loss to W&B
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {train_loss:.4f}")
        
        # Optionally save the model after each epoch
        if config['training']['save_model']:
            model_save_path = f"{config['training']['model_save_path']}model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)

    wandb.finish()

if __name__ == "__main__":
    main()
