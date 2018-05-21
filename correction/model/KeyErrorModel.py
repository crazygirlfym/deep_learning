#--*- coding:utf-8 -*--
from model.HmmModel import *
import numpy as np

elem_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def create_emission_matrix():
    ## 利用键盘的错误模拟一个概率  有些地方有高斯来模拟这个概率
    prob_t_c = 0.8
    prob_t_i = (1 - prob_t_c) / 2
    prob_d_c = 1 - prob_t_i
    prob_d_i = prob_t_i

    emit_matrix = {
        # '^': {'^': 1.0},
        'a': {'a': prob_d_c, 's': prob_d_i},
        'b': {'b': prob_t_c, 'v': prob_t_i, 'n': prob_t_i},
        'c': {'c': prob_t_c, 'x': prob_t_i, 'v': prob_t_i},
        'd': {'d': prob_t_c, 's': prob_t_i, 'f': prob_t_i},
        'e': {'e': prob_t_c, 'w': prob_t_i, 'r': prob_t_i},
        'f': {'f': prob_t_c, 'd': prob_t_i, 'g': prob_t_i},
        'g': {'g': prob_t_c, 'f': prob_t_i, 'h': prob_t_i},
        'h': {'h': prob_t_c, 'g': prob_t_i, 'j': prob_t_i},
        'i': {'i': prob_t_c, 'u': prob_t_i, 'o': prob_t_i},
        'j': {'j': prob_t_c, 'h': prob_t_i, 'k': prob_t_i},
        'k': {'k': prob_t_c, 'j': prob_t_i, 'l': prob_t_i},
        'l': {'l': prob_d_c, 'k': prob_d_i},
        'm': {'m': prob_d_c, 'n': prob_d_i},
        'n': {'n': prob_t_c, 'b': prob_t_i, 'm': prob_t_i},
        'o': {'o': prob_t_c, 'i': prob_t_i, 'p': prob_t_i},
        'p': {'p': prob_d_c, 'o': prob_d_i},
        'q': {'q': prob_d_c, 'w': prob_d_i},
        'r': {'r': prob_t_c, 'e': prob_t_i, 't': prob_t_i},
        's': {'s': prob_t_c, 'a': prob_t_i, 'd': prob_t_i},
        't': {'t': prob_t_c, 'r': prob_t_i, 'y': prob_t_i},
        'u': {'u': prob_t_c, 'y': prob_t_i, 'i': prob_t_i},
        'v': {'v': prob_t_c, 'c': prob_t_i, 'b': prob_t_i},
        'w': {'w': prob_t_c, 'q': prob_t_i, 'e': prob_t_i},
        'x': {'x': prob_t_c, 'z': prob_t_i, 'c': prob_t_i},
        'y': {'y': prob_t_c, 't': prob_t_i, 'u': prob_t_i},
        'z': {'z': prob_d_c, 'x': prob_d_i}
        # '$': {'$': 1.0}
    }

    size = len(emit_matrix)
    emission_matrix = np.full((size, size), 0 , dtype=float)
    elem_list.sort()
    # print(elem_list)
    keys_list = elem_list
    # print(len(keys_list))
    map_to_zero = list(zip(keys_list, range(0, size)))
    map_to_zero = dict(map_to_zero)
    # print(map_to_zero)
    for key in emit_matrix:
        for letter, probability in emit_matrix[key].items():
            i = map_to_zero[key]
            j = map_to_zero[letter]
            emission_matrix[i,j] = probability


    result = np.matrix(emission_matrix)

    observation = keys_list

    return result, observation, map_to_zero

def create_transition_matrix(filepath, map_to_zero):
    start_probability = {}
    trans_matrix = {}

    with open(filepath) as f:
        total_word=0
        for line in f:
            line = line.strip()
            seg = line.split(' ')
            chn_char = seg[0]
            if len(chn_char) > 1:
                break

            freq = float(seg[1])
            pinyin = seg[3]

            last_char = '^'

            # trans_matrix.setdefault(last_char, {})
            for ch, index in zip(pinyin, range(len(pinyin))):
                start_probability.setdefault(ch, 0)
                if last_char == '^':
                    # print("test")
                    total_word += 1
                    if (ch in start_probability):
                        # print (start_probability)
                        start_probability[ch] = start_probability[ch] + 1
                    else:
                        start_probability[ch] = 1
                    last_char = ch
                    continue
                trans_matrix.setdefault(last_char, {})
                trans_matrix[last_char].setdefault(ch, 0)
                trans_matrix[last_char][ch] = trans_matrix[last_char][ch] + 1
                # trans_matrix[last_char][ch] = trans_matrix.get(last_char, {}).get(ch, 0.0) + 1
                last_char = ch



    start_probability_matrix = [0 for i in range (len(map_to_zero))]
    # for key, value in start_probability.items():
    # print(total_word)

    for elem in elem_list:
        if elem not in start_probability:
            start_probability_matrix[map_to_zero[elem]]  = (float(1) /(total_word + 26))

        else:
        ## 需要做一个平滑
            start_probability_matrix[map_to_zero[elem]] = (float(1+start_probability[elem]) /(total_word + 26))


    transition_matrix = np.full((len(map_to_zero), len(map_to_zero)), 0, dtype=float)


    for elem in elem_list:
        if elem not in trans_matrix:
            for elem2 in elem_list:
                transition_matrix[map_to_zero[elem]][map_to_zero[elem2]] = 1.0 / 26

        else:

            count = sum(trans_matrix[elem].values()) + 26
            for elem2 in elem_list:
                if elem2 not in trans_matrix[elem]:
                    transition_matrix[map_to_zero[elem]][map_to_zero[elem2]] = (1.0 / count)
                else:
                    transition_matrix[map_to_zero[elem]][map_to_zero[elem2]] = (1.0 + trans_matrix[elem][elem2]) / count

    return np.matrix(transition_matrix), np.matrix(start_probability_matrix)

if __name__ == "__main__":
    emission_matrix , observation, map_to_zero= create_emission_matrix()
    # print(map_to_zero)
    filename = "../data/googlepinyin.txt"
    transition_matrix, start_probability_matrix = create_transition_matrix(filename, map_to_zero)

    observation=elem_list

    mispelling_modes = MispinyinHMM(start_probability=start_probability_matrix, transition_matrix=transition_matrix,
                                    hidden_states=observation, observables= observation, emission_matrix=emission_matrix)
    # ## maybe zhongguo
    # print(observation)
    # print(transition_matrix)start_probability_matrix
    obs1 = ('m', 'o','a','n')
    x = mispelling_modes.viterbi(obs1)
    print(x)





