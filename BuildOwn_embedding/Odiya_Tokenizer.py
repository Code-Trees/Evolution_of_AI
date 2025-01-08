from collections import Counter
from tqdm import tqdm
import numpy as np
import pickle

class Tokenizer_BPE:
    def __init__(self, text, vocab_size=5000):
        self.max_vocab_size = vocab_size
        self.corpus = text

    def _get_token_stats(self, ids):
        """Compute frequency of adjacent token pairs."""
        return Counter(zip(ids, ids[1:]))

    def _merge_tokens(self, ids, pair, new_idx):
        """Merge occurrences of the most frequent pair."""
        i = 0
        merged_ids = []
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged_ids.append(new_idx)
                i += 2
            else:
                merged_ids.append(ids[i])
                i += 1
        return merged_ids

    def train(self):
        """Train the Byte Pair Encoding tokenizer."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        num_merges = self.max_vocab_size - 256

        # Encode the corpus into byte tokens.
        ids = list(self.corpus.encode("utf-8"))
        tokens = ids.copy()
        pbar = tqdm(range(num_merges), desc="Training BPE Tokenizer")
        for _ in pbar:
            stats = self._get_token_stats(ids)
            if not stats:
                break

            # Select the most frequent pair.
            most_frequent_pair = max(stats, key=stats.get)

            # Assign a new index to the pair and merge.
            new_idx = len(self.vocab)
            ids = self._merge_tokens(ids, most_frequent_pair, new_idx)

            # Update vocab and merges.
            self.merges[most_frequent_pair] = new_idx
            self.vocab[new_idx] = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            pbar.set_description(f"Iteration {_}, Compression Ratio {len(tokens) / len(ids):.2f}X")
            if _ in np.arange( 0,num_merges,num_merges/1000):
                print(f"Crossed {len(tokens) / len(ids):.2f} ")
                
        print("++++++++++++++++++++++++++++Final Result ++++++++++++++++++++++++++++")
        print(f"After training: tokens length: {len(ids)}")
        print(f"After training: merges length: {len(self.merges)}")
        print(f"After Training Vocab length {len(self.vocab)}")
        print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

        return self.vocab, self.merges

    def encode(self, text):
        """Encode text into BPE tokens."""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_token_stats(tokens)
            if not stats:
                break

            # Find the next pair to merge.
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break

            new_idx = self.merges[pair]
            tokens = self._merge_tokens(tokens, pair, new_idx)

        return tokens

    def decode(self, ids):
        """Decode BPE tokens back to text."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, filepath):
        """Save the tokenizer to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({"vocab": self.vocab, "merges": self.merges}, f)
        print(f"Tokenizer saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load the tokenizer from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        tokenizer = Tokenizer_BPE("")
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data["merges"]
        return tokenizer