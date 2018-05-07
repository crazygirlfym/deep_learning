# --*- coding:utf-8 -*--
import codecs
import numpy as np
def read_data(file_path):
    text = codecs.open(file_path, 'r', 'utf-8', errors="ignore").read()
    chars = list(set(text))
    text_size, vocab_size = len(text), len(chars)
    print ('text has %d characters, %d unique' % (text_size, vocab_size))
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}
    return char2id, id2char, text


def characters(probabilities, id2char):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char[c] for c in np.argmax(probabilities, 1)]

def batches2string(batches, id2char):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b, id2char))]
    return s