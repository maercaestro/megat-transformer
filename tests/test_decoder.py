# test_decoder.py - Tests for the Decoder and DecoderLayer classes

import unittest
import torch
from src.decoder import Decoder, DecoderLayer

class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        # Set up parameters for DecoderLayer
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.vocab_size = 10000  # Define vocab_size here
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate DecoderLayer
        self.decoder_layer = DecoderLayer(self.d_model, self.h, self.d_ff, self.dropout)
    
    def test_decoder_layer_forward(self):
        # Set up mock input tensors
        x = torch.rand(self.batch_size, self.sequence_length, self.d_model)
        enc_output = torch.rand(self.batch_size, self.sequence_length, self.d_model)
        
        # Updated mask shapes to match attention scores dimensions
        tgt_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)
        memory_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)

        # Run forward pass
        output = self.decoder_layer(x, enc_output, tgt_mask, memory_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for DecoderLayer")

# Similarly for TestDecoder
class TestDecoder(unittest.TestCase):
    def setUp(self):
        # Set up parameters for Decoder
        self.num_layers = 6
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.vocab_size = 10000  # Define vocab_size here
        self.max_len = 512
        self.dropout = 0.1
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate Decoder
        self.decoder = Decoder(self.num_layers, self.d_model, self.h, self.d_ff, self.vocab_size, self.max_len, self.dropout)
    
    def test_decoder_instantiation(self):
        # Check that the decoder initializes with the expected submodules
        self.assertIsInstance(self.decoder.input_embedding, torch.nn.Module, "Expected input_embedding to be an instance of nn.Module")
        self.assertIsInstance(self.decoder.pos_encoding, torch.nn.Module, "Expected pos_encoding to be an instance of nn.Module")
        self.assertIsInstance(self.decoder.norm, torch.nn.LayerNorm, "Expected norm to be an instance of nn.LayerNorm")
        self.assertEqual(len(self.decoder.layers), self.num_layers, f"Expected {self.num_layers} layers in Decoder")

    def test_decoder_forward(self):
        # Set up mock input tensors
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))  # Input token indices
        enc_output = torch.rand(self.batch_size, self.sequence_length, self.d_model)
        
        # Updated mask shapes to match attention scores dimensions
        tgt_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)
        memory_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)

        # Run forward pass
        output = self.decoder(x, enc_output, tgt_mask, memory_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for Decoder")

if __name__ == '__main__':
    unittest.main()
