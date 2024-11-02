# custom_dataset.py - Custom Dataset class without tokenizer

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, vocab, max_len):
        """
        Custom dataset for loading CSV data without using padding or unknown tokens.

        Args:
            data_path (str): Path to the CSV file.
            vocab (dict): Vocabulary mapping words/characters to IDs.
            max_len (int): Maximum sequence length for truncation.
        """
        # Load the data and filter out empty sequences
        data = pd.read_csv(data_path)
        filtered_data = data[data["source_text"].str.strip().astype(bool) & data["translated_text"].str.strip().astype(bool)]
        self.data = filtered_data.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get source and target text
        src_text = self.data.iloc[idx]["source_text"]
        tgt_text = self.data.iloc[idx]["translated_text"]

        # Convert to vocab IDs and truncate to max_len
        src_ids = self.text_to_ids(src_text)
        tgt_ids = self.text_to_ids(tgt_text)

        # Handle cases where either src_ids or tgt_ids is empty
        if not src_ids:
            src_ids = [0] * self.max_len  # Placeholder sequence
        if not tgt_ids:
            tgt_ids = [0] * self.max_len  # Placeholder sequence

        return {
            'input': torch.tensor(src_ids, dtype=torch.long),
            'target': torch.tensor(tgt_ids, dtype=torch.long)
        }

    def text_to_ids(self, text):
        """
        Convert text to a list of vocab IDs and truncate to max_len.
        """
        ids = [self.vocab[token] for token in text.split() if token in self.vocab]
        # Truncate to max_len (no padding)
        return ids[:self.max_len]

class VocabularyBuilder:
    def __init__(self, data_path):
        """
        Initializes the VocabularyBuilder with the path to the dataset.

        Args:
            data_path (str): Path to the CSV file containing source and target texts.
        """
        self.data_path = data_path
        self.vocab = {}     # Initialize an empty vocabulary
        self.max_len = 0    # Maximum sequence length

    def build(self):
        """
        Builds the vocabulary and calculates the maximum sequence length.

        Returns:
            tuple: (vocab, max_len), where vocab is a dictionary mapping each unique token to a unique ID,
                   and max_len is an integer representing the longest sequence length in the dataset.
        """
        data = pd.read_csv(self.data_path)

        for _, row in data.iterrows():
            for text in [row["source_text"], row["translated_text"]]:
                tokens = text.split()
                self.max_len = max(self.max_len, len(tokens))  # Update max_len for the longest sequence
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)  # Add token to vocab with a unique ID

        return self.vocab, self.max_len

