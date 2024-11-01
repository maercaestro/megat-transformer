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
