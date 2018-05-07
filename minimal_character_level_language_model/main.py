# --*- coding:utf-8 -*--
from utils import *
from DataGenerator import BatchGenerator
from min_char_lstm import *
if __name__ == '__main__':
    char2id, id2char, train_text = read_data('./input.txt')
    batch_size = 16
    num_unrollings = 30
    vocabulary_size = len(char2id)
    generator = BatchGenerator(train_text, batch_size, num_unrollings, vocabulary_size, char2id,)

    text = batches2string(generator.next(), id2char)
    for i in text:
        print(i)

    model = SimpleModel(vocabulary_size, 5000, generator)
    model.run()
