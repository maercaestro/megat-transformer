# train.py - Training script for Transformer model using YAML configuration

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.transformer import Transformer
from config.config import load_config  # Import YAML config loader
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
    config = load_config()  # Assumes config.yaml is in the same directory
    
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
    
    # Load training data
    train_dataset = CustomDataset(data_path=config['data']['train_data_path'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {train_loss:.4f}")
        
        # Optionally save the model after each epoch
        if config['training']['save_model']:
            model_save_path = f"{config['training']['model_save_path']}model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()
