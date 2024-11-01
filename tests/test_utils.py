# test_utils.py - Tests for the utility classes in utils.py

import unittest
import torch
from src.utils import InputEmbedding, PositionalEncoding, FeedForwardNetwork

class TestInputEmbedding(unittest.TestCase):
    def setUp(self):
        # Define parameters for InputEmbedding
        self.vocab_size = 10000
        self.d_model = 512
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate InputEmbedding
        self.input_embedding = InputEmbedding(self.vocab_size, self.d_model)

    def test_input_embedding_instantiation(self):
        # Check that the embedding layer is properly initialized
        self.assertIsInstance(self.input_embedding.embedding, torch.nn.Embedding, "Expected embedding to be an instance of nn.Embedding")

    def test_input_embedding_forward(self):
        # Create a mock input tensor with token indices
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))

        # Run forward pass
        output = self.input_embedding(x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for InputEmbedding")


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        # Define parameters for PositionalEncoding
        self.d_model = 512
        self.max_len = 512
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate PositionalEncoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_len)

    def test_positional_encoding_instantiation(self):
        # Check that the positional encoding buffer is registered correctly
        self.assertTrue(hasattr(self.positional_encoding, 'pe'), "Expected positional encoding buffer 'pe' to be registered")

    def test_positional_encoding_forward(self):
        # Create a mock input tensor
        x = torch.rand(self.sequence_length, self.batch_size, self.d_model)

        # Run forward pass
        output = self.positional_encoding(x)

        # Check output shape
        self.assertEqual(output.shape, (self.sequence_length, self.batch_size, self.d_model), "Unexpected output shape for PositionalEncoding")


class TestFeedForwardNetwork(unittest.TestCase):
    def setUp(self):
        # Define parameters for FeedForwardNetwork
        self.d_model = 512
        self.d_ff = 2048
        self.sequence_length = 10
        self.batch_size = 2

        # Instantiate FeedForwardNetwork
        self.ffn = FeedForwardNetwork(self.d_model, self.d_ff)

    def test_feedforward_network_instantiation(self):
        # Check that the linear layers and dropout are properly initialized
        self.assertIsInstance(self.ffn.linear1, torch.nn.Linear, "Expected linear1 to be an instance of nn.Linear")
        self.assertIsInstance(self.ffn.linear2, torch.nn.Linear, "Expected linear2 to be an instance of nn.Linear")
        self.assertIsInstance(self.ffn.relu, torch.nn.ReLU, "Expected relu to be an instance of nn.ReLU")
        self.assertIsInstance(self.ffn.dropout, torch.nn.Dropout, "Expected dropout to be an instance of nn.Dropout")

    def test_feedforward_network_forward(self):
        # Create a mock input tensor
        x = torch.rand(self.batch_size, self.sequence_length, self.d_model)

        # Run forward pass
        output = self.ffn(x)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.sequence_length, self.d_model), "Unexpected output shape for FeedForwardNetwork")

if __name__ == '__main__':
    unittest.main()
