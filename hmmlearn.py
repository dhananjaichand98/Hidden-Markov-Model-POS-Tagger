import json
import sys
import math
from collections import defaultdict

class HmmTrainer:
    """
        This class trains hmm model on tagged input text file and stores weights to output file

        Attributes:
            initial_probs: log of initial probabilities for states
            final_probs: log of probability of a state to be final state for a sequence of observations
            states: set of states
            obsvs: set of observations
            state_freq: frequencey of occurance of each state
            state_freq_non_end: frequence of occurance of each state except for occurance as last word
            transition_log_probs: transition log probabilities map for pairs of previous state and current state
            emission_log_probs: emission log probabilities map for pairs of state and observations
            open_class_states: states belonging to open class
            uniques_obsvs_per_state: map of states to frequency of unique observations
            suffix_emission_log_probs: emission log probabilities map for pairs of state and suffixes
            suffixes: set of suffixes seen on training data
            suffixes_count: count of occurance of suffixes
    """

    def __init__(self) -> None:
        """
            Inits HmmTrainer
        """
        self.initial_probs = {}
        self.final_probs = {}
        self.states = set()
        self.obsvs = set()
        self.state_freq = {}
        self.state_freq_non_end = {}
        self.transition_log_probs = {}
        self.emission_log_probs = {}
        self.open_class_states = set()
        self.uniques_obsvs_per_state = {}
        self.suffix_emission_log_probs = {}
        self.suffixes = set()
        self.suffixes_count = defaultdict(lambda: 0)

    def add_state(self, state) -> None:
        """
            add state to states dict
        """
        self.states.add(state)

    def add_obsvs(self, obsv) -> None:
        """
            add observation to obsv dict
        """
        self.obsvs.add(obsv)

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

                if (state, obsv) in self.emission_log_probs:
                    self.emission_log_probs[(state, obsv)] += 1
                else:
                    self.emission_log_probs[(state, obsv)] = 1

                    if state in self.uniques_obsvs_per_state:
                        self.uniques_obsvs_per_state[state] += 1
                    else:
                        self.uniques_obsvs_per_state[state] = 1

                suffix = obsv[-2:]
                if len(suffix) == 2 and len(obsv) > 2:
                    if (state, suffix) in self.suffix_emission_log_probs:
                        self.suffix_emission_log_probs[(state, suffix)] += 1
                    else:
                        self.suffix_emission_log_probs[(state, suffix)] = 1
                    self.suffixes_count[suffix] += 1

                suffix = obsv[-3:]
                if len(suffix) == 3 and len(obsv) > 3:
                    if (state, suffix) in self.suffix_emission_log_probs:
                        self.suffix_emission_log_probs[(state, suffix)] += 1
                    else:
                        self.suffix_emission_log_probs[(state, suffix)] = 1
                    self.suffixes_count[suffix] += 1

                if i > 0:
                    prev_state = tokens[i-1][0]
                    if (prev_state, state) in self.transition_log_probs:
                        self.transition_log_probs[(prev_state, state)] += 1
                    else:
                        self.transition_log_probs[(prev_state, state)] = 1

                if i != len(tokens) - 1:
                    if state in self.state_freq_non_end:
                        self.state_freq_non_end[state] += 1
                    else:
                        self.state_freq_non_end[state] = 1
                else:
                    if state in self.final_probs:
                        self.final_probs[state] += 1
                    else:
                        self.final_probs[state] = 1

    def train(self, parsed_lines, smooth_transition=True, smooth_emission=False, smooth_initial_probs=True)  -> None:
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
                if not ((state, next_state) in self.transition_log_probs):
                    self.transition_log_probs[(state, next_state)] = 1 if smooth_transition else 0
                else:
                    self.transition_log_probs[(state, next_state)] += 1 if smooth_transition else 0
                state_transition_total[state] += self.transition_log_probs[(state, next_state)]

            state_observation_total[state] = 0
            for obsv in self.obsvs:
                if not ((state, obsv) in self.emission_log_probs):
                    self.emission_log_probs[(state, obsv)] = 1 if smooth_emission else 0
                else:
                    self.emission_log_probs[(state, obsv)] += 1 if smooth_emission else 0
                state_observation_total[state] += self.emission_log_probs[(state, obsv)]

            for suffix in self.suffixes_count.keys():
                if not ((state, suffix) in self.suffix_emission_log_probs):
                    self.suffix_emission_log_probs[(state, suffix)] = 1
                else:
                    self.suffix_emission_log_probs[(state, suffix)] += 1

            if not (state in self.initial_probs):
                self.initial_probs[state] = 0
            
            if not (state in self.final_probs):
                self.final_probs[state] = 0

        # counting total number of states
        total_states = sum(self.state_freq.values())

        state_alpha = 1
        obsv_alpha = 1
        # calculating log probabilities for tranisition and emission pairs
        for state in self.states:
            for next_state in self.states:
                self.transition_log_probs[(state, next_state)] = state_alpha*self.transition_log_probs[(state, next_state)]/(self.state_freq[state] + 5*len(self.states))  + (1-state_alpha)*self.state_freq[next_state]/total_states
                self.transition_log_probs[(state, next_state)] = math.log(self.transition_log_probs[(state, next_state)]) if self.transition_log_probs[(state, next_state)] != 0 else float("-inf") 
            for obsv in self.obsvs:
                self.emission_log_probs[(state, obsv)] = obsv_alpha*self.emission_log_probs[(state, obsv)]/(state_observation_total[state]) + (1-obsv_alpha)*self.state_freq[state]/total_states
                self.emission_log_probs[(state, obsv)] = math.log(self.emission_log_probs[(state, obsv)]) if self.emission_log_probs[(state, obsv)] != 0 else float("-inf")

        # calculating log probabilities for initial states
        total_initial_P_count = 0
        for key, value in self.initial_probs.items():
            self.initial_probs[key] += 1 if smooth_initial_probs else 0
            total_initial_P_count += self.initial_probs[key]

        for key, value in self.initial_probs.items():
            self.initial_probs[key] = (value)/total_initial_P_count
            self.initial_probs[key] = math.log(self.initial_probs[key]) if self.initial_probs[key] != 0 else float("-inf")

        # calculating log probabilities for final states
        total_final_P_count = 0
        smooth_final_probs = True
        for key, value in self.final_probs.items():
            self.final_probs[key] += 1 if smooth_final_probs else 0
            total_final_P_count += self.final_probs[key]

        for key, value in self.final_probs.items():
            self.final_probs[key] = (value)/total_final_P_count
            self.final_probs[key] = math.log(self.final_probs[key]) if self.final_probs[key] != 0 else float("-inf")

        # calculating log probabilities for suffix emission pairs
        for key, value in self.suffix_emission_log_probs.items():
            self.suffix_emission_log_probs[key] = value/self.state_freq[key[0]]
            self.suffix_emission_log_probs[key] - math.log(self.suffix_emission_log_probs[key]) if self.suffix_emission_log_probs[key] != 0 else float("-inf")

        # storing the top suffixes        
        sorted_suffixes = sorted(self.suffixes_count.items(), key=lambda item: item[1], reverse=True)
        self.suffixes = [x[0] for x in sorted_suffixes[:100]]

        # setting open class states
        sorted_uniq_obsvs_states = sorted(self.uniques_obsvs_per_state.items(), key=lambda item: item[1], reverse=True)

        sorted_uniq_obsvs_states = sorted_uniq_obsvs_states[:5]
        for [state, _] in sorted_uniq_obsvs_states:
            self.open_class_states.add(state)

    def save_to_file(self, file) -> None:
        """
            save weights to output file
        """
        json_dict = {}
        json_dict["initial_probs"] = self.initial_probs
        json_dict["final_probs"] = self.final_probs
        json_dict["transition_log_probs"] = dict(
            (' '.join(k), v) for k, v in self.transition_log_probs.items())
        json_dict["emission_log_probs"] = dict(
            (' '.join(k), v) for k, v in self.emission_log_probs.items())
        json_dict["suffix_emission_log_probs"] = dict(
            (' '.join(k), v) for k, v in self.suffix_emission_log_probs.items()) 
        json_dict["state_freq"] = self.state_freq
        json_dict["open_class_states"] = list(self.open_class_states)
        json_dict["suffixes"] = list(self.suffixes)
        with open(file, 'w', encoding='utf-8') as fp:
            json.dump(json_dict, fp, indent=2, ensure_ascii=False)

def main():
    ht = HmmTrainer()
    lines = []
    try:
        input_file = sys.argv[1]
    except:
        input_file = "data/it_isdt_train_tagged.txt"
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            lines.append(line)
    parsed_lines = ht.parse_input(lines)
    ht.train(parsed_lines)
    ht.save_to_file('hmmmodel.txt')


if __name__ == "__main__":
    main()