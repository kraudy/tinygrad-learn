words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
"""Get alphabet."""
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
"""Letters to int. 28 Elements"""
itos = {i:s for s,i in stoi.items()}

import numpy as np
import torch

