import hidden_markov

class MispinyinHMM():
    def __init__(self, hidden_states, observables, start_probability, transition_matrix, emission_matrix):
        self.hidden_states = hidden_states
        self.observables = observables
        self.start_probability = start_probability
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self._model_ = hidden_markov.hmm(states=hidden_states, observations=observables,
                                         start_prob=start_probability, trans_prob=transition_matrix,
                                         em_prob=emission_matrix)

    def viterbi(self, sequence):
        return self._model_.viterbi(sequence)

    def forward(self, sequence):
        return self._model_.forward_algo(sequence)




