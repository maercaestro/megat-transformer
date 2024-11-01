# evaluate.py - Evaluation script for Transformer model

import torch
from torch.utils.data import DataLoader
from src.transformer import Transformer
from dataset.custom_dataset import CustomDataset
from config.config import load_config

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            
            outputs = model(inputs, targets)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model based on config parameters
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
    
    # Load the trained model weights
    model.load_state_dict(torch.load(config['evaluation']['model_load_path']))
    model.eval()
    
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Load the evaluation dataset
    # Sample vocabulary - Replace with actual vocabulary for your dataset
    vocab = {
        "hi": 1,
        "dunia": 2,
        "<pad>": 0,
        "<unk>": 3
        # Add other words or characters as needed
    }
    eval_dataset = CustomDataset(
        data_path=config['data']['eval_data_path'],
        vocab=vocab,
        max_len=config['model']['max_len']
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['evaluation']['batch_size'])
    
    # Run evaluation
    eval_loss = evaluate(model, eval_dataloader, criterion, device)
    print(f"Evaluation Loss: {eval_loss:.4f}")

if __name__ == "__main__":
    main()
