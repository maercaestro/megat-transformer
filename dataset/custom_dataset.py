# custom_dataset.py - Custom Dataset class with VocabularyBuilder

import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
import csv

class VocabularyBuilder:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vocab = {}
        self.max_len = 0

    def build(self):
        try:
            # Use on_bad_lines='skip' to ignore problematic rows
            data = pd.read_csv(self.data_path, on_bad_lines='skip')
        except pd.errors.ParserError:
            print("Parser error encountered in the CSV file.")
            return self.vocab, self.max_len

        # Process the data as before...
        for _, row in data.iterrows():
            for text in [row["source_text"], row["translated_text"]]:
                tokens = text.split()
                self.max_len = max(self.max_len, len(tokens))
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
        return self.vocab, self.max_len


class CustomDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_len=128):
        """
        Custom dataset for loading CSV data and handling vocabulary building.

        Args:
            data_path (str): Path to the CSV file.
            vocab (dict or None): Pre-built vocabulary mapping words/characters to IDs.
            max_len (int): Maximum sequence length for padding/truncation.
        """
        self.data = pd.read_csv(data_path)
        self.max_len = max_len

        # If no vocab is provided, build it from the dataset
        if vocab is None:
            vocab_builder = VocabularyBuilder(min_freq=1)
            all_texts = pd.concat([self.data["source_text"], self.data["translated_text"]]).tolist()
            self.vocab = vocab_builder.build(all_texts)
        else:
            self.vocab = vocab

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
