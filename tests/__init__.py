# test_init.py - Test for __init__.py

import unittest
from src import (
    Transformer,
    Encoder,
    Decoder,
    ScaledDotProductAttention,
    MultiHeadAttention,
    InputEmbedding,
    PositionalEncoding,
    FeedForwardNetwork
)

class TestInit(unittest.TestCase):
    
    def test_imports(self):
        # Check that each module is correctly imported
        self.assertIsNotNone(Transformer, "Transformer should be imported from src.")
        self.assertIsNotNone(Encoder, "Encoder should be imported from src.")
        self.assertIsNotNone(Decoder, "Decoder should be imported from src.")
        self.assertIsNotNone(ScaledDotProductAttention, "ScaledDotProductAttention should be imported from src.")
        self.assertIsNotNone(MultiHeadAttention, "MultiHeadAttention should be imported from src.")
        self.assertIsNotNone(InputEmbedding, "InputEmbedding should be imported from src.")
        self.assertIsNotNone(PositionalEncoding, "PositionalEncoding should be imported from src.")
        self.assertIsNotNone(FeedForwardNetwork, "FeedForwardNetwork should be imported from src.")
        
        # Optionally, check that each is a class
        self.assertTrue(isinstance(Transformer, type), "Transformer should be a class.")
        self.assertTrue(isinstance(Encoder, type), "Encoder should be a class.")
        self.assertTrue(isinstance(Decoder, type), "Decoder should be a class.")
        self.assertTrue(isinstance(ScaledDotProductAttention, type), "ScaledDotProductAttention should be a class.")
        self.assertTrue(isinstance(MultiHeadAttention, type), "MultiHeadAttention should be a class.")
        self.assertTrue(isinstance(InputEmbedding, type), "InputEmbedding should be a class.")
        self.assertTrue(isinstance(PositionalEncoding, type), "PositionalEncoding should be a class.")
        self.assertTrue(isinstance(FeedForwardNetwork, type), "FeedForwardNetwork should be a class.")

if __name__ == '__main__':
    unittest.main()
