from collections import Counter
from .BasicTokenizer import BasicTokenizer
import regex as re

# TODO: Improve specifically for Dante
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
  def __init__(self, pattern=None):
    super().__init__()
    self.pattern = pattern if pattern else GPT4_PATTERN

  def train(self, text, vocab_size, verbose=False):
    num_to_merge = vocab_size - 256

    # split the text up into text chunks
    text_chunks = re.findall(self.pattern, text)

    # input text preprocessing
    tokenized_text = [list(chunk.encode("utf-8")) for chunk in text_chunks]
    _, self.merges, self.rev_merges = self.continuous_merge(tokenized_text, num_to_merge)

  def count_bigrams(self, tokenized_text):
    return Counter(bigram for word in tokenized_text for bigram in zip(word[:-1], word[1:]))

  def continuous_merge(self, tokenized_text, n_to_merge):
      tokenized_text = tokenized_text.copy()  # Work on a copy to leave the original unchanged.

      # Compute a base index that is outside the valid UTF-8 range.
      base = 256
      merges = {}
      rev_merges = {}

      for merge_iteration in range(n_to_merge):
          counter = self.count_bigrams(tokenized_text=tokenized_text)
          if not counter:
              break
          # Get the most frequent bigram
          bigram_to_merge, count = max(counter.items(), key=lambda item: item[1])
          new_idx = base + merge_iteration
          tokenized_text = [self.merge(tokenized_word, bigram_to_merge, new_idx) for tokenized_word in tokenized_text]
          merges[bigram_to_merge] = new_idx
          rev_merges[new_idx] = bigram_to_merge
      return tokenized_text, merges, rev_merges