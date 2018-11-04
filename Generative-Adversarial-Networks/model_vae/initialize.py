
import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, contant=1):
    low = -contant * np.sqrt(6.0 /(fan_in + fan_out))
    high = contant * np.sqrt(6.0 /(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


