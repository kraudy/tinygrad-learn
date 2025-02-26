words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(f"len: {len(words)}")
print(f"min len word: {min(len(w) for w in words)}")
print(f"max len word: {max(len(w) for w in words)}")

b = {}
for w in words[:3]:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1

print(b)

from tinygrad import Tensor, nn, TinyJit, dtypes

N = Tensor.empty(28, 28, dtype=dtypes.int32)
N[1,3] = 1
print(N.numpy())