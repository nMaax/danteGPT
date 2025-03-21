from collections import Counter

class BasicTokenizer():
  def __init__(self,):
    self.merges = None
    self.rev_merges = None
    self.vocab = None

  def train(self, text, vocab_size, verbose=False):

    assert vocab_size > 256, "vocab_size must be greater than 256"
    num_to_merge = vocab_size - 256
    tokens = list(map(int, text.encode('utf-8')))
    _, self.merges, self.rev_merges = self.continuous_merge(tokens, num_to_merge)

  def encode(self, text):
    merges=self.merges
    tokens = list(text.encode('utf-8'))  # Get byte representation
    for bigram, new_idx in merges.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == bigram:
                new_tokens.append(new_idx)  # Merge the bigram
                i += 2  # Skip next token (merged)
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens  # Update tokens after each merge pass
    return tokens

  def decode(self, tokens):
    """Decodes merged tokens back into a UTF-8 string."""

    vocab = {token: bytes([token]) for token in range(256)}
    for (original_token_1, original_token_2), merged_token in self.merges.items():
        vocab[merged_token] = vocab[original_token_1] + vocab[original_token_2]
    self.vocab = vocab

    tokens = b"".join(vocab[token] for token in tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text

  def count_bigrams(self, tokens):
    return Counter(zip(tokens[:-1], tokens[1:]))

  def sort_counter(self, counter, reverse=True):
      return sorted(((count, bigram) for bigram, count in counter.items()), reverse=reverse)

  def merge(self, tokens, bigram, new_idx):
      new_tokens = []
      i = 0
      while i < len(tokens):
          if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == bigram:
              new_tokens.append(new_idx)
              i += 2  # Skip the next token as it's merged
          else:
              new_tokens.append(tokens[i])
              i += 1
      return new_tokens

  def continuous_merge(self, tokens, n_to_merge):
      tokens = tokens.copy()  # Work on a copy to leave the original unchanged.
      # Compute a base index that is outside the valid UTF-8 range.
      base = 256
      merges = {}
      rev_merges = {}

      for merge_iteration in range(n_to_merge):
          counter = self.count_bigrams(tokens)
          if not counter:
              break
          sorted_counter = self.sort_counter(counter, reverse=True)
          # Get the most frequent bigram
          bigram_to_merge = sorted_counter[0][1]
          new_idx = base + merge_iteration
          tokens = self.merge(tokens, bigram_to_merge, new_idx)
          merges[bigram_to_merge] = new_idx
          rev_merges[new_idx] = bigram_to_merge
      return tokens, merges, rev_merges