import numpy as np
import tensorflow as tf
def padding_sequence(self, inputs, max_length=None):

    batch_size = len(inputs)

    if max_length != None:
        if np.max([len(i) for i in inputs]) > max_length:
            maxlen = max_length
        else:
            maxlen = np.max([len(i) for i in inputs])

    output = np.zeros([batch_size, maxlen], dtype=np.int32)

    for i, seq in enumerate(inputs):
        output[i, :len(seq[: maxlen])] = np.array(seq[:maxlen])
    return output, maxlen

def Mask(inputs, seq_len, mode ='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len))
        for _ in range(len(inputs.shape) -2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12