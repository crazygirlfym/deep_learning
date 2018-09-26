#--*- coding:utf-8 -*--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def generate_sample(file, context_size):

    with open(file) as f:
        for line in f:
            line = line.strip()
            index_words = line.split()
            for index, center in enumerate(index_words):
                for target in index_words[max(0, index - context_size): index]:
                    yield center, target
                for target in index_words[index + 1: index + context_size + 1]:
                    yield center, target


def gen(batch_size, context_size, file):
    single_gen = generate_sample(file, context_size)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch