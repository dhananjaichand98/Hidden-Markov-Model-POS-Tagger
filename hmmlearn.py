import json
import numpy as np
import sys
import math

class HmmTrainer:
    
    def __init__(self) -> None:
        self.initialP = {}
        self.states = {}
        self.obsvs = {}
        # for training
        self.state_freq = {}
        self.state_freq_non_end = {}
        self.transition_probs = {}
        self.emission_probs = {}
    
    def add_state(self, state):
        if not (state in self.states):
            self.states[state] = len(self.states)

    def add_obsvs(self, obsv):
        if not (obsv in self.obsvs):
            self.obsvs[obsv] = len(self.obsvs)

    def parse_input(self, lines):
        parsed_lines = []
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

    def initialize(self, parsed_lines):

        for tokens in parsed_lines:
            if(len(tokens) == 0):
                continue

            first_state, _ = tokens[0]

            if first_state in self.initialP:
                self.initialP[first_state] += 1
            else:
                self.initialP[first_state]  = 1 

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

        # print("initial probabilities of states: ", json.dumps(self.initialP, indent = 4))
        # print("number of states : ", len(self.states))
        # print("number of observations : ", len(self.obsvs))
        # print("States dict:", self.states)
        # print("States freq:", self.state_freq)
        # print("States freq non end:", self.state_freq_non_end)
    
    def train(self):

        state_transition_total = {}
        state_observation_total = {}

        for state in self.states:
            state_transition_total[state] = 0
            for next_state in self.states:
                if not((state, next_state) in self.transition_probs):
                    self.transition_probs[(state, next_state)] = 1
                else:
                    self.transition_probs[(state, next_state)] += 1
                state_transition_total[state] += self.transition_probs[(state, next_state)]
        
            state_observation_total[state] = 0
            for obsv in self.obsvs:
                if not((state, obsv) in self.emission_probs):
                    self.emission_probs[(state, obsv)] = 1 
                else:
                    self.emission_probs[(state, obsv)] += 1
                state_observation_total[state] += self.emission_probs[(state, obsv)]

            if not (state in self.initialP):
                self.initialP[state] = 0 # for smoothing

        # ROW WISE SMOOTHING FO STATE TRANSITION AND EMISSIONS
        for state in self.states:
            for next_state in self.states:
                self.transition_probs[(state, next_state)] /= state_transition_total[state]

            running_sum = 0
            for obsv in self.obsvs:
                self.emission_probs[(state, obsv)] /= state_observation_total[state]
                running_sum += self.emission_probs[(state, obsv)]

        # SMOOTHING AND GETTING INITIAL PROBABILITIES
        total_initial_P_count = 0
        for key, value in self.initialP.items():
            self.initialP[key] = (value + 1)
            total_initial_P_count += self.initialP[key]
        
        for key, value in self.initialP.items():
            self.initialP[key] = (value)/total_initial_P_count

    def save_to_file(self, file):
        json_dict = {}
        json_dict["initialP"] = self.initialP
        json_dict["transition_probs"] = dict((' '.join(k), v) for k, v in self.transition_probs.items())
        json_dict["emission_probs"] = dict((' '.join(k), v) for k, v in self.emission_probs.items())
        json_dict["state_freq"] = self.state_freq
        with open(file, 'w', encoding='utf8') as fp:
            json.dump(json_dict, fp)

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
    ht.initialize(parsed_lines)
    ht.train() 
    ht.save_to_file('data/hmmmodel.txt')       

if __name__ == "__main__":
    main()