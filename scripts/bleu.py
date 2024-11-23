import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from dataset.custom_dataset import CustomDataset, BuildVocabulary
from src.transformer import Transformer
from config.config import load_config
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import wandb

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
config = load_config()

# Load full dataset
dataset_path = config["data"]["dataset_path"]
df = pd.read_csv(dataset_path)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize vocabulary and test dataset
source_vocab = BuildVocabulary(df['source_text'].tolist())
target_vocab = BuildVocabulary(df['translated_text'].tolist())
test_dataset = CustomDataset(test_df, source_vocab, target_vocab, config["data"]["source_max_len"], config["data"]["target_max_len"])

# Initialize model and move to GPU if available
model = Transformer(
    num_layers=config["model"]["num_layers"],
    d_model=config["model"]["d_model"],
    h=config["model"]["num_heads"],
    d_ff=config["model"]["d_ff"],
    src_vocab_size=len(source_vocab.vocab),
    tgt_vocab_size=len(target_vocab.vocab),
    max_len=max(config["data"]["source_max_len"], config["data"]["target_max_len"]),
    dropout=config["model"].get("dropout_rate", 0.1)
).to(device)  # Move model to GPU if available

# Load latest checkpoint and extract model weights
checkpoint_path = "/latest_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)  # Load to appropriate device
model.load_state_dict(checkpoint["model_state_dict"])  # Load only the model weights
model.eval()

# Helper function to generate translations
def generate_translation(model, src_seq, src_vocab, tgt_vocab, max_len=50):
    src_tensor = torch.tensor([src_vocab.text_to_sequence(src_seq)]).long().to(device)  # Move to GPU if available
    tgt_seq = [0]  # Start with a dummy token to initialize

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_seq]).long().to(device)  # Ensure tgt_tensor is on the same device
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
            next_token = output.argmax(-1)[:, -1].item()  # Greedy decoding
            tgt_seq.append(next_token)

    # Convert generated indices back to tokens
    rev_vocab = {idx: token for token, idx in tgt_vocab.vocab.items()}
    return " ".join([rev_vocab[idx] for idx in tgt_seq if idx in rev_vocab])

# BLEU evaluation function
def evaluate_bleu(model, dataset, src_vocab, tgt_vocab):
    total_bleu = 0
    rev_src_vocab = {idx: token for token, idx in src_vocab.vocab.items()}
    rev_tgt_vocab = {idx: token for token, idx in tgt_vocab.vocab.items()}
    smooth_fn = SmoothingFunction().method1
    
    for i in range(len(dataset)):
        source_seq = dataset[i]["source_seq"]
        target_seq = dataset[i]["target_seq"]
        
        # Convert source and target sequences back to text
        src_seq_text = " ".join([rev_src_vocab[idx.item()] for idx in source_seq if idx.item() != src_vocab.vocab["<pad>"]])
        ref_text = " ".join([rev_tgt_vocab[idx.item()] for idx in target_seq if idx.item() != tgt_vocab.vocab["<pad>"]])

        # Generate translation
        translation = generate_translation(model, src_seq_text, src_vocab, tgt_vocab).split()
        reference = ref_text.split()
        
        # Calculate BLEU for this sample with smoothing
        total_bleu += sentence_bleu([reference], translation, smoothing_function=smooth_fn)

    avg_bleu = total_bleu / len(dataset)
    print(f"Average BLEU Score: {avg_bleu:.4f}")

# Run BLEU evaluation
evaluate_bleu(model, test_dataset, source_vocab, target_vocab)
