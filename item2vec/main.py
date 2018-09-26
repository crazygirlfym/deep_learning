## -*-- coding:utf-8 -*--

from utils.data_utils import gen
from model.item2vec import Item2Vec
import tensorflow as tf
from utils.process import get_movie_name_id_dict
from utils.process import get_movie_id_name_dict
from findneightbor import load_vectors
from findneightbor import topk_like
class Config(object):
    batch_size = None
    context_size = None
    file = None

def train_gen():
    yield from gen(batch_size=Config.batch_size, context_size=Config.context_size, file=Config.file)

def cal_vocab_size(file):
    vocab_set = set()
    with open(file) as f:
        for line in f:
            line = line.strip()
            vocab = line.split()
            for item in vocab:
                vocab_set.add(item)
    return len(vocab_set)

def main():
    embed_size = 200
    batch_size = 100
    context_size = 5
    num_sample = 10
    learning_rate = 0.01
    file = "./data/doulist_0804_09.movie_id"
    Config.batch_size = batch_size
    Config.context_size = context_size
    Config.file = file
    vocab_size = cal_vocab_size(file)
    # Config.index_to_word = index_to_word
    # Config.index_of_words = index_of_words
    # print(len(index_of_words))
    # print(index_of_words[0])
    # print(len(index_of_words[1]))

    dataset = tf.data.Dataset.from_generator(train_gen, (tf.int32, tf.int32),
                                             (tf.TensorShape([batch_size]), tf.TensorShape([batch_size, 1])))
    model = Item2Vec(dataset=dataset, vocab_size=vocab_size, embed_size=embed_size, batch_size=batch_size,
                          num_sample=num_sample,  learning_rate=learning_rate)
    model.build_graph()
    model.train()
    #model_save_path = "./checkpoint"
    #model.load_vector(model_save_path, "./data/save_vector.vec")

def similarity_test():
    movie_name_id_dict = get_movie_name_id_dict()
    movie_id_name_dict = get_movie_id_name_dict()
    vectors = load_vectors("./data/save_vector.vec")
    movie_names = ['小时代', '倩女幽魂', '悟空传', '美国往事', '战狼2']
    for movie_name  in movie_names:
        res= topk_like(movie_name_id_dict, movie_id_name_dict, vectors, movie_name, print_log=False)
        print("the similar movies for %s are as follow" %movie_name)
        for item in res:
            print(movie_id_name_dict[item[0]])
        print("-------")

if __name__ == '__main__':

    ## follow http://www.diqiuzhuanzhuan.com/2018/04/03/tensorflow%E5%AE%9E%E7%8E%B0word2vec/
    main()
    #similarity_test()
