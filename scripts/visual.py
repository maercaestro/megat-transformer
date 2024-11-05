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
checkpoint = torch.load(checkpoint_path)

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

# Visualization functions

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
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

def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn

def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(*charts).properties(title="Layer %d" % (layer + 1))

# Visualize attention for two specific rows
def visualize_attention_for_selected_rows(selected_rows):
    charts = []
    for idx, row in selected_rows.iterrows():
        src_text = row['source_text']
        tgt_text = row['translated_text']

        # Tokenize the source and target sequences
        src_tokens = src_text.split()
        tgt_tokens = tgt_text.split()

        # Visualize encoder attention for Layer 0
        chart = visualize_layer(
            model, 
            layer=0, 
            getter_fn=get_encoder, 
            ntokens=len(src_tokens), 
            row_tokens=src_tokens, 
            col_tokens=src_tokens
        )
        chart = chart.properties(title=f"Attention for Row {idx}")
        charts.append(chart)
        
    return alt.vconcat(*charts)

# Run visualization for the selected rows
attention_charts = visualize_attention_for_selected_rows(selected_rows)
attention_charts.display()  # Display the chart in a notebook environment
