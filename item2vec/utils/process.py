#--*- coding:utf-8 -*--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from collections import Counter



DoulistFile = './data/doulist_0804_09.json'
MovieFile = './data/movie_0804_09.json'
DoulistCorpusIdFile = DoulistFile.replace('json', 'movie_id')
DoulistCorpusNameFile = DoulistFile.replace('json', 'movie_name')


def shuffle(reader, buf_size):
    """
    create shuffle reader with the size of buf_size
    :param reader: the original reader to be shuffled
    :param buf_size: shuffle buffer size
    :return: the new reader
    """

    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)

        if len(buf) >= buf_size:
            random.shuffle(buf)
            for b in buf:
                yield  b
            buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader()

def get_movie_name_id_dict(file=DoulistFile, min_word_freq=0):
    movie_counter = Counter()
    with open(file) as f:
        for line in f:
            line = line.strip()
            json_str = json.loads(line)
            # for movie_name in json_str['movie_names']:
            movie_counter.update(json_str['movie_names'])
    movie_freq = filter(lambda  e: e[1] >= min_word_freq, movie_counter.items())
    movie_counter_sorted = sorted(movie_freq, key=lambda x: (-x[1], x[0]))
    movies, _ = list(zip(*movie_counter_sorted))
    movie_name_id_dict = dict(zip(movies, range(len(movies))))
    movie_name_id_dict['<unk>'] = len(movies)
    print('movie_name_id_dict is %d from [%s]' % (len(movie_name_id_dict), file))
    return movie_name_id_dict

def get_movie_id_name_dict(doulist_file=DoulistFile):
    movie_name_id_dict = get_movie_name_id_dict(doulist_file)
    movie_id_name_dict = dict([(_[1], _[0]) for _ in movie_name_id_dict.items()])
    print('movie_id_name_dict is %d from [%s]' % (len(movie_id_name_dict), doulist_file))
    return movie_id_name_dict

def process2corpus():
    movie_name_id_dict = get_movie_name_id_dict()
    print(movie_name_id_dict)
    print('total movie is %d from [%s], [%s]' % (len(movie_name_id_dict), DoulistFile, MovieFile))
    with open(DoulistFile) as fopen, open(DoulistCorpusNameFile, 'w') as fwrite, open(DoulistCorpusIdFile,
                                                                                      'w') as fwrite_1:
        for line in fopen:
            doulist_dict = json.loads(line.strip())
            doulist_movies =doulist_dict['movie_names']
            doulist_movie_ids = [str(movie_name_id_dict[_]) for _ in doulist_movies]
            fwrite.write('%s\n' % ('\t'.join(doulist_movies)))
            fwrite_1.write('%s\n' % (' '.join(doulist_movie_ids)))

def main():
    process2corpus()

if __name__== "__main__":
    main()