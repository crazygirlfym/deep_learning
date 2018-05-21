## -*-- coding:utf-8 -*--

from model.HmmModel import *
import numpy as np
elem_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n','o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
def parseSentence(fileObj):
    word_list = []
    for line in fileObj:
        line  = line.strip()
        for word in line.split():
            word_list.append(word)
    return word_list

def create_all_matrix(correct_corpus_path, with_wrong_corpus_path):
    training_file = open(correct_corpus_path, 'r+') ## 正确的文本
    training_crpt_file = open(with_wrong_corpus_path, 'r+') ##包含错误的文本

    training_words = parseSentence(training_file)
    training_crpt_words = parseSentence(training_crpt_file)

    start_probab = {}
    trans_probab = {}
    emission_probab = {}
    total_word = 0
    for i in range(0, len(training_words)):
        ch_first = training_words[i][0]
        if not ch_first.isalpha():
            continue

        start_probab.setdefault(ch_first, 0)
        start_probab[ch_first] = start_probab[ch_first] + 1
        total_word += 1
        ch_orig = ch_first
        ch_crpt = training_crpt_words[i][0]
        for j in range(1, len(training_words[i])):
            ch_orig_next = training_words[i][j]
            ch_crpt_next = training_crpt_words[i][j]
            if not ch_orig_next.isalpha():
                continue
            trans_probab.setdefault(ch_orig, {})
            trans_probab[ch_orig].setdefault(ch_orig_next, 0)
            trans_probab[ch_orig][ch_orig_next] = trans_probab[ch_orig][ch_orig_next] + 1
            if not ch_crpt_next.isalpha():
                continue

            emission_probab.setdefault(ch_orig, {})
            emission_probab[ch_orig].setdefault(ch_crpt, 0)
            emission_probab[ch_orig][ch_crpt] = emission_probab[ch_orig][ch_crpt] + 1

            ch_orig = ch_orig_next
            ch_crpt = ch_crpt_next

    map_to_zero = list(zip(elem_list, range(0, len(elem_list))))
    map_to_zero = dict(map_to_zero)

    size = len(elem_list)
    emission_matrix = np.full((size, size), 0, dtype=float)
    start_matrix = [0 for i in range(size)]
    transition_matrix =  np.full((size, size), 0, dtype=float)

    for elem in elem_list:
        if elem not in start_probab:
            start_matrix[map_to_zero[elem]] = 1.0 / (total_word + 26)
        else:
            start_matrix[map_to_zero[elem]] = float(start_probab[elem] + 1)  / (total_word + 26)


    for elem_1 in elem_list:
        if elem_1 not in emission_probab:
            for elem_2 in elem_list:
                emission_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = 1.0 /26
        else:
            count = sum(emission_probab[elem_1].values())
            for elem_2 in elem_list:
                if elem_2 not in emission_probab[elem_1]:
                    emission_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = 1.0 / (count + 26)
                else:
                    emission_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = float(emission_probab[elem_1][elem_2] + 1) / (count + 26)



    for elem_1 in elem_list:
        if elem_1 not in trans_probab:
            for elem_2 in elem_list:
                transition_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = 1.0 /26
        else:
            count = sum(trans_probab[elem_1].values())
            for elem_2 in elem_list:
                if elem_2 not in trans_probab[elem_1]:
                    transition_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = 1.0 / (count + 26)
                else:
                    transition_matrix[map_to_zero[elem_1]][map_to_zero[elem_2]] = float(trans_probab[elem_1][elem_2] +1) / (count + 26)

    return np.matrix(start_matrix), np.matrix(transition_matrix), np.matrix(emission_matrix)






if __name__ == "__main__":
    correct_corpus_path = "../data/training.txt"
    with_wrong_corpus_path = "../data/training_corrupt.txt"
    start_matrix, transition_matrix, emission_matrix = create_all_matrix(correct_corpus_path, with_wrong_corpus_path)
    # print(start_matrix)
    # print(transition_matrix)
    # print(emission_matrix)
    observation = elem_list
    summation = np.sum(emission_matrix, axis=1)

    mispelling_modes = MispinyinHMM(start_probability=start_matrix, transition_matrix=transition_matrix,
                                    hidden_states=observation, observables= observation, emission_matrix=emission_matrix)
    # ## maybe zhongguo
    # print(observation)
    # print(transition_matrix)start_probability_matrix

    obs1 = ('i','n','d', 'u', 'w', 't', 'r', 'i', 'w','l')
    x = mispelling_modes.viterbi(obs1)
    print(x)
