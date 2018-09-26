# -*-- coding:utf-8 -*--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging
import sys
import heapq

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('item2vec')
class minHeap():
    def __init__(self, k):
        self._k = k
        self._heap = []

    def add(self, item):
        if len(self._heap) < self._k:
            self._heap.append(item)
            heapq.heapify(self._heap)
        else:
            if item > self._heap[0]:
                self._heap[0] = item
                heapq.heapify(self._heap)

    def get_min(self):
        if len(self._heap) > 0:
            return self._heap[0]
        else:
            return -2

    def get_all(self):
        return self._heap

def similarity(v1, v2):
    """
    :param v1:  vector 1, dimension is hidden layer number
    :param v2:  vector 2, dimension is hidden layer number
    :return:  similarity between v1 and v2
    """

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2)

def load_vectors(input_file):
    vectors = {}
    with open(input_file) as f:
        idx = 0
        for line in f:
            line = line.strip()
            line_list = line.split()

            vec = np.array([float(_) for _ in line_list], dtype=float)
            if not idx in vectors:
                vectors[idx] = vec
            idx += 1
    return vectors

def topk_like(movie_name_id_dict, movie_id_name_dict, vectors, cur_movie_name, k=5, print_log=False):
    
    # global movie_name_id_dict
    # global movie_id_name_dict
    # global vectors

    min_heap = minHeap(k)
    like_candidates = []
    if cur_movie_name not in movie_name_id_dict:
        logging.info('%s not in movie_name_id_dict' %cur_movie_name)
        return []

    cur_movie_id = movie_name_id_dict[cur_movie_name]
    if cur_movie_id not in vectors:
        logging.info('%s not in vectors' %cur_movie_id)
        return []

    cur_vec = vectors[cur_movie_id]
    if print_log:
        logging.info('[%d]%s top %d likes:' % (cur_movie_id, cur_movie_name, k))

    for movie_id, vec in vectors.items():
        if movie_id == cur_movie_id:
            continue

        sim = similarity(cur_vec, vec)
        if len(like_candidates) < k or sim > min_heap.get_min():
            min_heap.add(sim)
            like_candidates.append((movie_id, sim))
    if print_log:
        for t in sorted(like_candidates, reverse=True, key=lambda _: _[1])[:k]:
            logger.info('[%d]%s %f' % (t[0], movie_id_name_dict[t[0]], t[1]))
    return sorted(like_candidates, reverse=True, key=lambda _: _[1])[:k]






