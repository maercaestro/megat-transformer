# Megat Transformer

**Megat Transformer** is an implementation of the Transformer model, inspired by the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. This project is designed to provide a comprehensive and flexible Transformer architecture for a variety of NLP tasks, with modular components such as multi-head attention, positional encoding, and feed-forward networks.

## About the Paper

The Transformer model was introduced in 2017 by Vaswani et al., in the paper "Attention Is All You Need," which revolutionized the field of NLP. It introduced a novel self-attention mechanism to handle dependencies in long sequences, removing the need for recurrent or convolutional layers. The Transformer model has since become the foundation for state-of-the-art NLP models, such as BERT, GPT, and T5.

Key insights from the paper:
- **Self-Attention Mechanism**: A mechanism to capture dependencies in a sequence, allowing each token to attend to every other token in the sequence.
- **Positional Encoding**: Since there are no recurrent or convolutional layers, positional encodings are added to input embeddings to maintain the order of tokens.
- **Parallelization**: The model can be efficiently parallelized, which makes it faster to train on large datasets.

This project implements the original Transformer architecture as described in the paper.

## Project Structure

```plaintext
megat-transformer/
├── src/
│   ├── __init__.py               # Initializes the module imports
│   ├── encoder.py                # Encoder and EncoderLayer classes
│   ├── decoder.py                # Decoder and DecoderLayer classes
│   ├── transformer.py            # Main Transformer model
│   ├── attention.py              # Attention mechanisms (e.g., multi-head attention)
│   └── utils.py                  # Utility classes (e.g., embedding, positional encoding)
├── dataset/
│   └── combined_dataset.csv      # Example dataset (to be replaced with your own data)
├── scripts/
│   ├── train.py                  # Training script
│   └── evaluate.py               # Evaluation script
├── tests/
│   ├── test_encoder.py           # Tests for encoder components
│   ├── test_decoder.py           # Tests for decoder components
│   ├── test_transformer.py       # Tests for the Transformer model
│   └── test_utils.py             # Tests for utility classes
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

## Setup

To set up this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/megat-transformer.git
   cd megat-transformer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Weights & Biases for Experiment Tracking (optional)**:
   If you’re using [Weights & Biases (W&B)](https://wandb.ai/) to track experiments and save checkpoints, initialize W&B with your API key:
   ```bash
   wandb login
   ```

## Usage

### Training
Run the training script with:
```bash
python scripts/train.py
```
This will train the model on your dataset, specified in the `data/combined_dataset.csv` file. You can modify training configurations like batch size, learning rate, and epochs in the `config.yaml` file.

### Evaluation
To evaluate a trained model:
```bash
python scripts/evaluate.py
```
This will load a saved model checkpoint and evaluate its performance on the test dataset.

### Testing
To run tests on each component:
```bash
python -m unittest discover tests/
```

## Model Components

- **Encoder**: Processes the input sequence with a stack of encoder layers, each containing self-attention and feed-forward sub-layers.
- **Decoder**: Generates the target sequence by attending to both its own previous outputs and the encoder outputs.
- **Attention Mechanisms**: Includes scaled dot-product attention and multi-head attention.
- **Positional Encoding**: Adds positional information to input embeddings to maintain token order.
- **Feed-Forward Network**: Applies two linear transformations with a ReLU activation in between, a key part of each encoder and decoder layer.

## Checkpoints and Experiment Tracking

This project supports experiment tracking and checkpoint saving using [Weights & Biases (W&B)](https://wandb.ai/). Checkpoints are saved after each epoch, allowing you to resume training from any saved point. 

## References

- **Attention Is All You Need** by Vaswani et al. (2017): [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

With Megat Transformer, you can explore and experiment with the powerful Transformer architecture and customize it for various NLP tasks. Happy training!

