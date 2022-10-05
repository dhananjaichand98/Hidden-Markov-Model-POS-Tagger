import json
import sys

class HmmModel:

    def __init__(self) -> None:
        self.states = {}
        self.obsvs = {}
        self.initialP = {}
        # for training
        self.state_freq = {}
        self.state_freq_non_end = {}
        self.transition_probs = {}
        self.emission_probs = {}

    def load_weights(self, weights_file):
        with open(weights_file, 'r', encoding='utf8', errors='ignore') as f:
            json_dict = json.loads(f.read())
            self.initialP = json_dict["initialP"]
            self.state_freq = json_dict["state_freq"]
            self.transition_probs = dict((tuple(k.split()), v) for k, v in json_dict["transition_probs"].items())
            self.emission_probs = dict((tuple(k.split()), v) for k, v in json_dict["emission_probs"].items())
        
        # print("initial probabilities of states: ", json.dumps(self.initialP, indent = 4))
        # print("States freq:", self.state_freq)
        # print("size of transition probs:", len(self.transition_probs))
        # print("size of emission probs:", len(self.emission_probs))

    def parse_input(self, lines):
        parsed_lines = []
        for line in lines:
            tokens = line.split()
            parsed_line = []
            for token in tokens:
                obsv = token
                parsed_line.append(obsv)
            parsed_lines.append(parsed_line)
        return parsed_lines

    def run(self, lines):

        tagged_lines = []

        for line in lines:
            tags = self.viterbi_decoding(line)
            tagged_line = []
            # creating a tuple of obsv, state
            for i, obsv in enumerate(line):
                tagged_line.append((obsv, tags[i]))
            tagged_lines.append(tagged_line)

        return tagged_lines

    def viterbi_decoding(self, obsv_sequence):

        viterbi = {}
        tagged_line = []

        if (len(obsv_sequence) == 0):
            return tagged_line

        back_pointer = {}

        for state in self.initialP.keys():
            if(not ((state, obsv_sequence[0]) in self.emission_probs)):
                viterbi[(state, 0)] = self.initialP[state]
            else:
                viterbi[(state, 0)] = self.initialP[state] * self.emission_probs[(state, obsv_sequence[0])] 
            back_pointer[(state, 0)] = None

        for t in range(1, len(obsv_sequence)):
            for state in self.initialP.keys():
                curr, back = 0, None

                for prev_state in self.initialP.keys():
                    alpha = 0
                    if(not ((state, obsv_sequence[t]) in self.emission_probs)):
                        alpha = viterbi[(prev_state, t-1)] * self.transition_probs[(prev_state, state)]
                    else:
                        alpha = viterbi[(prev_state, t-1)] * self.transition_probs[(prev_state, state)] * self.emission_probs[(state, obsv_sequence[t])]                    
                    if alpha >= curr:
                        curr = alpha
                        back = prev_state                
                viterbi[(state, t)] = curr
                back_pointer[(state, t)] = back
        
        state_sequence = []
        bestpathprob = 0
        bestpathstate = None

        for state in self.initialP.keys():
            if viterbi[(state, len(obsv_sequence)-1)] > bestpathprob:
                bestpathprob = viterbi[(state, len(obsv_sequence)-1)]
                bestpathstate = state
        
        state_sequence.append(bestpathstate)
        prev_state = back_pointer[(bestpathstate, len(obsv_sequence)-1)]

        for t in range(len(obsv_sequence)-2, -1, -1):
            state_sequence.insert(0, prev_state)
            prev_state = back_pointer[(prev_state, t)]
        
        return state_sequence

    def write_output(self, file, tagged_lines):

        with open(file, 'w', encoding='utf8') as file:
            for tagged_line in tagged_lines:
                output_line = []
                for obsv, state in tagged_line:
                    output_line.append(f'{obsv}/{state}')
                file.write(' '.join(output_line))
                file.write('\n')

    # --------------------------------------------------------
    # FOR COMPARISON

    def parse_tagged_input(self, lines):
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

    def compare_result(self, file, tagged_pred_lines):
        lines = []
        with open(file, 'r', encoding='utf8', errors='ignore') as file:    
            for line in file:
                lines.append(line)
        parsed_test_lines = self.parse_tagged_input(lines)
        total, count = 0, 0
        for i in range(len(parsed_test_lines)):
            for j in range(len(parsed_test_lines[i])):
                _, pred_state = tagged_pred_lines[i][j]
                test_state, _ = parsed_test_lines[i][j]
                if(pred_state == test_state):
                    count += 1
                total += 1
        print(f'Accuracy: {count/total}')

def main():
    hm = HmmModel()
    hm.load_weights('hmmmodel.txt')
    try:
        input_file = sys.argv[1]
    except:
        input_file = "data/it_isdt_dev_raw.txt"

    lines = []
    with open(input_file, 'r', encoding='utf8', errors='ignore') as file:    
        for line in file:
            lines.append(line)
    
    parsed_lines = hm.parse_input(lines)
    tagged_lines = hm.run(parsed_lines)
    output_file = "data/hmmoutput.txt"
    hm.write_output(output_file, tagged_lines)
    # hm.compare_result('it_isdt_dev_tagged.txt', tagged_lines)   

if __name__ == "__main__":
    main()