# test_transformer.py - Tests for the Transformer class

import unittest
import torch
from src.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # Define parameters for Transformer
        self.num_layers = 6
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.src_vocab_size = 10000
        self.tgt_vocab_size = 10000
        self.max_len = 512
        self.dropout = 0.1
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate Transformer
        self.transformer = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.h,
            d_ff=self.d_ff,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            max_len=self.max_len,
            dropout=self.dropout
        )

    def test_transformer_instantiation(self):
        # Check that the model initializes with the expected submodules
        self.assertIsInstance(self.transformer.encoder, torch.nn.Module, "Expected encoder to be an instance of nn.Module")
        self.assertIsInstance(self.transformer.decoder, torch.nn.Module, "Expected decoder to be an instance of nn.Module")
        self.assertIsInstance(self.transformer.output_layer, torch.nn.Linear, "Expected output_layer to be an instance of nn.Linear")

    def test_transformer_forward(self):
        # Set up mock input tensors
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.sequence_length))  # Input token indices for source
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.sequence_length))  # Input token indices for target
        
        # Masks
        src_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)
        tgt_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)
        memory_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)

        # Run forward pass
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.tgt_vocab_size), "Unexpected output shape for Transformer")

if __name__ == '__main__':
    unittest.main()
