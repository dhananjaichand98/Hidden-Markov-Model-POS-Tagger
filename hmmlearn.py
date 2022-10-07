import json
import numpy as np
import sys
import math


class HmmTrainer:
    """
        This class trains hmm model on tagged input text file and stores weights to output file

        Attributes:
            initial_probs: initial probabilities for states
            states: set of states
            obsvs: set of observations
            state_freq: frequencey of occurance of each state
            state_freq_non_end: frequence of occurance of each state except for occurance as last word
            transition_probs: transition probabilities map for pairs of previous state and current state
            emission_probs: emission probabilities map for pairs of state and observations
    """

    def __init__(self) -> None:
        """
            Inits HmmTrainer
        """
        self.initial_probs = {}
        self.states = {}
        self.obsvs = {}
        self.state_freq = {}
        self.state_freq_non_end = {}
        self.transition_probs = {}
        self.emission_probs = {}

    def add_state(self, state) -> None:
        """
            add state to states dict
        """
        if not (state in self.states):
            self.states[state] = len(self.states)

    def add_obsvs(self, obsv) -> None:
        """
            add observation to obsv dict
        """
        if not (obsv in self.obsvs):
            self.obsvs[obsv] = len(self.obsvs)

    def parse_input(self, lines) -> list:
        """
            parse input text line by line and split each token into obsv and state
        """
        parsed_lines = []
        # splitting each line into tokens and further splitting tokens to (observation, state) pairs
        for line in lines:
            tokens = line.split()
            parsed_line = []
            for token in tokens:
                try:
                    [obsv, state] = token.rsplit('/', 1)
                except:
                    print("error on token : ",  token)
                    exit(0)
                parsed_line.append((state, obsv))
            parsed_lines.append(parsed_line)
        return parsed_lines

    def _calculate_occurances(self, parsed_lines) -> None:
        """
            initialize probability lookup maps for tranition and emission probabilities with frequency of occurances
        """
        for tokens in parsed_lines:
            if (len(tokens) == 0):
                continue

            first_state, _ = tokens[0]

            # adding to initial occurance of state
            if first_state in self.initial_probs:
                self.initial_probs[first_state] += 1
            else:
                self.initial_probs[first_state] = 1

            # calculating pair-wise occurances of (state, next_state) and (state, observation) pairs along with state frequencies
            for i, (state, obsv) in enumerate(tokens):
                self.add_state(state)
                self.add_obsvs(obsv)

                if state in self.state_freq:
                    self.state_freq[state] += 1
                else:
                    self.state_freq[state] = 1

                if (state, obsv) in self.emission_probs:
                    self.emission_probs[(state, obsv)] += 1
                else:
                    self.emission_probs[(state, obsv)] = 1

                if i > 0:
                    prev_state = tokens[i-1][0]
                    if (prev_state, state) in self.transition_probs:
                        self.transition_probs[(prev_state, state)] += 1
                    else:
                        self.transition_probs[(prev_state, state)] = 1

                if i != len(tokens) - 1:
                    if state in self.state_freq_non_end:
                        self.state_freq_non_end[state] += 1
                    else:
                        self.state_freq_non_end[state] = 1

        self.T = np.array([[0] * len(self.states)] * len(self.states))
        self.E = np.array([[0] * len(self.states)] * len(self.obsvs))

    def train(self, parsed_lines, smooth_transition=False, smooth_emission=False, smooth_initial_probs=False)  -> None:
        """
            set probability lookup maps for tranition and emission probabilities after smoothing (if passed to be true).
        """

        self._calculate_occurances(parsed_lines)

        state_transition_total = {}
        state_observation_total = {}

        # adding all pair-wise occurances of (state, next_state) and (state, observation) map and additionally adding 1 to pairs if smoothing passsed to be true
        for state in self.states:
            state_transition_total[state] = 0
            for next_state in self.states:
                if not ((state, next_state) in self.transition_probs):
                    self.transition_probs[(state, next_state)] = 1 if smooth_transition else 0
                else:
                    self.transition_probs[(state, next_state)] += 1 if smooth_transition else 0
                state_transition_total[state] += self.transition_probs[(state, next_state)]

            state_observation_total[state] = 0
            for obsv in self.obsvs:
                if not ((state, obsv) in self.emission_probs):
                    self.emission_probs[(state, obsv)] = 1 if smooth_emission else 0
                else:
                    self.emission_probs[(state, obsv)] += 1 if smooth_emission else 0
                state_observation_total[state] += self.emission_probs[(state, obsv)]

            if not (state in self.initial_probs):
                self.initial_probs[state] = 0

        # counting total number of states
        total_states = sum(self.state_freq.values())

        state_alpha = 1
        obsv_alpha = 1
        # state-wise smoothing for state tranition and emission probabilities
        for state in self.states:
            for next_state in self.states:
                self.transition_probs[(state, next_state)] = state_alpha*self.transition_probs[(state, next_state)]/(self.state_freq[state] + 5*len(self.states) - 1)  + (1-state_alpha)*self.state_freq[next_state]/total_states

            for obsv in self.obsvs:
                self.emission_probs[(state, obsv)] = obsv_alpha*self.emission_probs[(state, obsv)]/(state_observation_total[state]) + (1-obsv_alpha)*self.state_freq[state]/total_states

        # state-wise smoothing for initial probabilities
        total_initial_P_count = 0
        for key, value in self.initial_probs.items():
            self.initial_probs[key] = value
            self.initial_probs[key] += 1 if smooth_initial_probs else 0
            total_initial_P_count += self.initial_probs[key]

        for key, value in self.initial_probs.items():
            self.initial_probs[key] = (value)/total_initial_P_count

    def save_to_file(self, file) -> None:
        """
            save weights to output file
        """
        json_dict = {}
        json_dict["initial_probs"] = self.initial_probs
        json_dict["transition_probs"] = dict(
            (' '.join(k), v) for k, v in self.transition_probs.items())
        json_dict["emission_probs"] = dict(
            (' '.join(k), v) for k, v in self.emission_probs.items())
        json_dict["state_freq"] = self.state_freq
        with open(file, 'w', encoding='utf8') as fp:
            json.dump(json_dict, fp, indent=2)


def main():
    ht = HmmTrainer()
    lines = []
    try:
        input_file = sys.argv[1]
    except:
        input_file = "data/it_isdt_train_tagged.txt"
    with open(input_file, 'r', encoding='utf8', errors='ignore') as file:
        for line in file:
            lines.append(line)
    parsed_lines = ht.parse_input(lines)
    ht.train(parsed_lines)
    ht.save_to_file('hmmmodel.txt')


if __name__ == "__main__":
    main()
