import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
import numpy as np

class BuildVocabulary:
    def __init__(self, texts, reserved_tokens=["<pad>"]):
        self.vocab = self.build_vocab(texts, reserved_tokens)
        
    def build_vocab(self, texts, reserved_tokens):
        all_words = Counter(" ".join(texts).split())
        vocab = {word: idx for idx, (word, _) in enumerate(all_words.items(), start=len(reserved_tokens))}
        for idx, token in enumerate(reserved_tokens):
            vocab[token] = idx  # Add reserved tokens at the start
        return vocab

    def text_to_sequence(self, text):
        return [self.vocab.get(word, self.vocab.get("<unk>")) for word in text.split()]

class CustomDataset(Dataset):
    def __init__(self, df, source_vocab, target_vocab, source_max_len=37, target_max_len=43):
        self.df = df
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source_text = self.df.iloc[idx]['source_text']
        target_text = self.df.iloc[idx]['translated_text']

        # Convert text to sequences
        source_seq = self.source_vocab.text_to_sequence(source_text)
        target_seq = self.target_vocab.text_to_sequence(target_text)

        # Apply padding
        source_seq = source_seq[:self.source_max_len] + [self.source_vocab.vocab["<pad>"]] * max(0, self.source_max_len - len(source_seq))
        target_seq = target_seq[:self.target_max_len] + [self.target_vocab.vocab["<pad>"]] * max(0, self.target_max_len - len(target_seq))

        # Generate attention masks
        source_mask = [1 if token != self.source_vocab.vocab["<pad>"] else 0 for token in source_seq]
        target_mask = [1 if token != self.target_vocab.vocab["<pad>"] else 0 for token in target_seq]

        return {
            "source_seq": torch.tensor(source_seq, dtype=torch.long),
            "target_seq": torch.tensor(target_seq, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
        }
