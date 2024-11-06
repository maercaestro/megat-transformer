import pandas as pd
import torch
from dataset.custom_dataset import CustomDataset, BuildVocabulary
from src.transformer import Transformer
from config.config import load_config
import altair as alt

# Load configuration
config = load_config()

# Load dataset and select specific rows
dataset_path = config["data"]["dataset_path"]
df = pd.read_csv(dataset_path)
selected_rows = df.iloc[[0, 1]]  # Select the first two rows for visualization

# Initialize vocabulary
source_vocab = BuildVocabulary(df['source_text'].tolist())
target_vocab = BuildVocabulary(df['translated_text'].tolist())

# Initialize model
checkpoint_path = "/content/latest_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU or GPU as needed

model = Transformer(
    num_layers=config["model"]["num_layers"],
    d_model=config["model"]["d_model"],
    h=config["model"]["num_heads"],
    d_ff=config["model"]["d_ff"],
    src_vocab_size=len(source_vocab.vocab),
    tgt_vocab_size=len(target_vocab.vocab),
    max_len=max(config["data"]["source_max_len"], config["data"]["target_max_len"]),
    dropout=config["model"].get("dropout_rate", 0.1)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Hook function to capture attention weights (if available)
attention_weights = {}

def save_attention_weights(module, input, output):
    if isinstance(output, tuple) and len(output) > 1:
        attention_weights["value"] = output[1]  # Assuming the second output is attention weights
    else:
        attention_weights["value"] = output  # Directly assign output if not a tuple

    # Log the shape of attention weights for debugging
    if attention_weights["value"] is not None:
        print(f"Captured attention weights shape: {attention_weights['value'].shape}")

# Register the hook on the encoder's self-attention layer (layer 0 for this example)
layer_to_hook = 0  # Adjust as needed to target other layers
model.encoder.layers[layer_to_hook].self_attn.register_forward_hook(save_attention_weights)

# Visualization functions
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    if m.ndim < 2:
        raise ValueError("Attention weights matrix has an unexpected shape.")
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    if attn is None or attn.ndim < 3:
        raise ValueError("Attention weights are missing or have an unexpected shape.")
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

# Visualize attention for selected rows
def visualize_attention_for_selected_rows(selected_rows):
    charts = []
    for idx, row in selected_rows.iterrows():
        src_text = row['source_text']
        tgt_text = row['translated_text']

        # Tokenize the source and target sequences
        src_tokens = src_text.split()
        tgt_tokens = tgt_text.split()

        # Convert tokens to tensors
        src_seq = torch.tensor([source_vocab.text_to_sequence(src_text)]).long()
        tgt_seq = torch.tensor([target_vocab.text_to_sequence(tgt_text)]).long()

        # Forward pass to capture attention weights
        model(src_seq, tgt_seq)  # This triggers the hook

        # Ensure attention weights are available and have the expected shape
        attn = attention_weights.get("value")
        if attn is None or attn.ndim < 3:
            print(f"Warning: Attention weights for Row {idx} are missing or have an unexpected shape.")
            continue

        # Generate attention map for each head
        n_heads = attn.shape[1]  # Number of attention heads
        charts.append(
            alt.vconcat(*[
                attn_map(attn, layer=0, head=h, row_tokens=src_tokens, col_tokens=src_tokens, max_dim=len(src_tokens))
                for h in range(n_heads)
            ]).properties(title=f"Attention for Row {idx}")
        )
        
    return alt.vconcat(*charts)

# Run visualization for the selected rows
try:
    attention_charts = visualize_attention_for_selected_rows(selected_rows)
    attention_charts.display()  # Display the chart in a notebook environment
except ValueError as e:
    print(f"Error during visualization: {e}")
