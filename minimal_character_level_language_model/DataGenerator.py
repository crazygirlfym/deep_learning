# --*- coding:utf-8 -*--
import numpy as np

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings, vocabulary_size, char2id, id2char):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        segment = self._text_size // batch_size
        self._vocabulary_size = vocabulary_size
        self._num_unrollings = num_unrollings
        self._char2id = char2id
        self._id2char = id2char
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()


    def _next_batch(self):

        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

        for b in range(self._batch_size):
            batch[b, self._char2id[self._text[self._cursor[b]]]] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size

        return batch

    def next(self):

        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
