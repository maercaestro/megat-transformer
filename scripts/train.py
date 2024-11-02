import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
from sklearn.model_selection import train_test_split
from src.transformer import Transformer
from config.config import load_config
from dataset.custom_dataset import CustomDataset  
from dataset.custom_dataset import VocabularyBuilder  # Import the VocabularyBuilder class

def pad_collate_fn(batch):
    """
    Collate function that pads sequences in each batch to the maximum length
    within the batch for proper batching.

    Args:
        batch (list of dicts): Each item is a dictionary with 'input' and 'target'.

    Returns:
        dict: A dictionary containing padded 'input' and 'target' tensors.
    """
    # Extract input and target sequences
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Determine the maximum length in this batch
    max_len = max([len(seq) for seq in inputs + targets])
    
    # Use 0 as the pad token (make sure this matches the vocabulary)
    pad_token_id = 0

    # Pad each input and target to the maximum length
    padded_inputs = [torch.cat([seq, torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)]) for seq in inputs]
    padded_targets = [torch.cat([seq, torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)]) for seq in targets]
    
    # Stack into a batch
    return {
        'input': torch.stack(padded_inputs),
        'target': torch.stack(padded_targets)
    }

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

        # Print iteration loss
        print(f"Epoch [{epoch+1}], Iteration [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Log batch loss to W&B
        wandb.log({"batch_loss": loss.item()})

    return total_loss / len(dataloader)

def main():
    # Load configuration from YAML file
    config = load_config()
    
    # Initialize Weights & Biases (W&B)
    wandb.init(
        project="megat-transformer",
        config=config,
        name="Transformer Training on 30,000 Rows",
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build vocabulary and get max_len from data_path
    vocab_builder = VocabularyBuilder(config['data']['data_path'])
    vocab, max_len = vocab_builder.build()
    
    # Calculate vocab size
    vocab_size = len(vocab)  # Use this for src_vocab_size and tgt_vocab_size

    # Initialize model with parameters from config.yaml
    model = Transformer(
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        h=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        src_vocab_size=vocab_size,  # Set the vocab size here
        tgt_vocab_size=vocab_size,  # Set the vocab size here
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Initialize dataset
    dataset = CustomDataset(data_path=config['data']['data_path'], vocab=vocab, max_len=max_len)
    
    # Select a subset of 300,000 rows and split into training and validation
    subset_indices = list(range(30000))
    train_indices, val_indices = train_test_split(subset_indices, test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Use the collate_fn argument for padding within batches
    train_dataloader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=pad_collate_fn)
    val_dataloader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=pad_collate_fn)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch)
        
        # Log training loss to W&B
        wandb.log({"epoch": epoch+1, "train_loss": train_loss})
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Average Loss: {train_loss:.4f}")
        
        # Optionally save the model after each epoch to W&B and local storage
        if config['training']['save_model']:
            model_save_path = f"{config['training']['model_save_path']}model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)  # Save checkpoint to W&B for future access

    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()
