# test_encoder.py - Tests for the Encoder and EncoderLayer classes

import unittest
import torch
from src.encoder import Encoder, EncoderLayer

class TestEncoderLayer(unittest.TestCase):
    def setUp(self):
        # Parameters for EncoderLayer
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate EncoderLayer
        self.encoder_layer = EncoderLayer(self.d_model, self.h, self.d_ff, self.dropout)

    def test_encoder_layer_instantiation(self):
        # Check that the layer initializes with the expected submodules
        self.assertIsInstance(self.encoder_layer.self_attn, torch.nn.Module, "Expected self_attn to be an instance of nn.Module")
        self.assertIsInstance(self.encoder_layer.ffn, torch.nn.Module, "Expected ffn to be an instance of nn.Module")
        self.assertEqual(len(self.encoder_layer.norm_layers), 2, "Expected two LayerNorm layers")
        self.assertEqual(len(self.encoder_layer.dropout), 2, "Expected two Dropout layers")
    
    def test_encoder_layer_forward(self):
        # Set up mock input tensor and mask
        x = torch.rand(self.batch_size, self.sequence_length, self.d_model)
        src_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)

        # Run forward pass
        output = self.encoder_layer(x, src_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for EncoderLayer")


class TestEncoder(unittest.TestCase):
    def setUp(self):
        # Parameters for Encoder
        self.num_layers = 6
        self.d_model = 512
        self.h = 8
        self.d_ff = 2048
        self.vocab_size = 10000
        self.max_len = 512
        self.dropout = 0.1
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate Encoder
        self.encoder = Encoder(self.num_layers, self.d_model, self.h, self.d_ff, self.vocab_size, self.max_len, self.dropout)

    def test_encoder_instantiation(self):
        # Check that the encoder initializes with the expected submodules
        self.assertIsInstance(self.encoder.input_embedding, torch.nn.Module, "Expected input_embedding to be an instance of nn.Module")
        self.assertIsInstance(self.encoder.pos_encoding, torch.nn.Module, "Expected pos_encoding to be an instance of nn.Module")
        self.assertIsInstance(self.encoder.norm, torch.nn.LayerNorm, "Expected norm to be an instance of nn.LayerNorm")
        self.assertEqual(len(self.encoder.layers), self.num_layers, f"Expected {self.num_layers} layers in Encoder")

    def test_encoder_forward(self):
        # Set up mock input tensor and mask
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))  # Input token indices
        src_mask = torch.ones(self.batch_size, self.h, self.sequence_length, self.sequence_length)

        # Run forward pass
        output = self.encoder(x, src_mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for Encoder")

if __name__ == '__main__':
    unittest.main()
