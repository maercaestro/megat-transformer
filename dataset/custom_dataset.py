# custom_dataset.py - Custom Dataset class without tokenizer

import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, vocab, max_len):
        """
        Custom dataset for loading CSV data without using a tokenizer.

        Args:
            data_path (str): Path to the CSV file.
            vocab (dict): Vocabulary mapping words/characters to IDs.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        self.data = pd.read_csv(data_path)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get source and target text
        src_text = self.data.iloc[idx]["source_text"]
        tgt_text = self.data.iloc[idx]["translated_text"]

        # Convert to vocab IDs and pad/truncate to max_len
        src_ids = self.text_to_ids(src_text)
        tgt_ids = self.text_to_ids(tgt_text)
        
        return {
            'input': torch.tensor(src_ids, dtype=torch.long),
            'target': torch.tensor(tgt_ids, dtype=torch.long)
        }

    def text_to_ids(self, text):
        """
        Convert text to a list of vocab IDs and pad/truncate to max_len.
        """
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in text.split()]
        # Pad or truncate to max_len
        if len(ids) < self.max_len:
            ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids
    
import pandas as pd

class VocabularyBuilder:
    def __init__(self, data_path):
        """
        Initializes the VocabularyBuilder with the path to the dataset.
        
        Args:
            data_path (str): Path to the CSV file containing source and target texts.
        """
        self.data_path = data_path
        self.vocab = {"<pad>": 0, "<unk>": 1}  # Start with special tokens
        self.max_len = 0

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
                        self.vocab[token] = len(self.vocab)

        return self.vocab, self.max_len

    def get_vocab(self):
        """Returns the generated vocabulary dictionary."""
        return self.vocab

    def get_max_len(self):
        """Returns the maximum sequence length found in the dataset."""
        return self.max_len
