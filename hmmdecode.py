import json
import sys

class HmmModel:
    """
        This class loads input weights and runs hmm model on raw text files

        Attributes:
            initial_probs: initial probabilities for states
            final_probs: log of probability of a state to be final state for a sequence of observations
            training_states: set of training states
            training_obsvs: set of training observations
            state_freq: frequencey of occurance of each state
            transition_log_probs: transition log probabilities map for pairs of previous state and current state
            emission_log_probs: emission log probabilities map for pairs of state and observations
            open_class_states: states belonging to open class
            suffix_emission_log_probs: emission log probabilities map for pairs of state and suffixes
            suffixes: map of suffixes with frequency of occurance, seen on training data
    """

    def __init__(self) -> None:
        """
            Inits HmmModel
        """
        self.initial_probs = {}
        self.training_states = {}
        self.training_obsvs = {}
        self.emission_log_probs = {}
        self.state_freq = {}
        self.transition_log_probs = {}
        self.open_class_states = set()
        self.suffix_emission_log_probs = {}
        self.final_probs = {}
        self.suffixes = {}

    def load_weights(self, weights_file) -> None:
        """
            load trained weights into class variables
        """
        with open(weights_file, 'r', encoding='utf-8', errors='ignore') as f:
            json_dict = json.loads(f.read())
            self.initial_probs = json_dict["initial_probs"]
            self.state_freq = json_dict["state_freq"]
            self.transition_log_probs = dict(
                (tuple(k.split()), v) for k, v in json_dict["transition_log_probs"].items())
            self.emission_log_probs = dict((tuple(k.split()), v)
                                       for k, v in json_dict["emission_log_probs"].items())
            self.suffix_emission_log_probs = dict((tuple(k.split()), v)
                                       for k, v in json_dict["suffix_emission_log_probs"].items())
            self.training_obsvs = set(k[1] for k in self.emission_log_probs.keys())
            self.training_states = set(self.initial_probs.keys())
            self.open_class_states = set(json_dict["open_class_states"])
            self.final_probs = json_dict["final_probs"]
            self.suffixes = json_dict["suffixes"]

    def parse_input(self, lines) -> list:
        """
            parse input text line by line and split into list of obsvations
        """
        parsed_lines = []
        # splitting each line into observations
        for line in lines:
            tokens = line.split()
            parsed_line = []
            for token in tokens:
                obsv = token
                parsed_line.append(obsv)
            parsed_lines.append(parsed_line)
        return parsed_lines

    def run(self, lines) -> list:
        """
            decode input line by line
        """
        tagged_lines = []
        # calling viterbi decoding algorithm for each line to get tags
        for line in lines:
            tags = self.viterbi_decoding(line)
            tagged_line = []
            # creating a tuple of obsv, state
            for i, obsv in enumerate(line):
                tagged_line.append((obsv, tags[i]))

            tagged_lines.append(tagged_line)

        return tagged_lines

    def viterbi_decoding(self, obsv_sequence) -> list:
        """
            decode a list of observations and tag them using viterbi decoding algorithm
        """

        viterbi = {}
        tagged_line = []

        if (len(obsv_sequence) == 0):
            return tagged_line

        back_pointer = {}

        initial_suffix_bi, initial_suffix_tri = obsv_sequence[0][-2:], obsv_sequence[0][-3:]        
        initial_suffix_first, initial_suffix_second = initial_suffix_bi, initial_suffix_tri

        if (initial_suffix_bi in self.suffixes and initial_suffix_tri in self.suffixes and self.suffixes[initial_suffix_tri] > self.suffixes[initial_suffix_bi]):
            initial_suffix_first, initial_suffix_second = initial_suffix_tri, initial_suffix_bi

        # calculating viterbi path log probability and back pointer for initial set of states (time t = 0)        
        for state in self.training_states:
            if (not ((state, obsv_sequence[0]) in self.emission_log_probs)):
                if obsv_sequence[0] in self.training_obsvs or self.initial_probs[state] == float("-inf") or not(state in self.open_class_states):
                    viterbi[(state, 0)] = float("-inf")
                else:
                    if initial_suffix_first in self.suffixes:
                        viterbi[(state, 0)] = self.initial_probs[state] + self.suffix_emission_log_probs[(state, initial_suffix_first)]
                    elif initial_suffix_second in self.suffixes:
                        viterbi[(state, 0)] = self.initial_probs[state] + self.suffix_emission_log_probs[(state, initial_suffix_second)]
                    else:
                        viterbi[(state, 0)] = self.initial_probs[state]
            else:
                if self.emission_log_probs[(state, obsv_sequence[0])] == float("-inf") or self.initial_probs[state] == float("-inf"):
                    viterbi[(state, 0)] = float("-inf")
                else:
                    viterbi[(state, 0)] = self.initial_probs[state] + self.emission_log_probs[(state, obsv_sequence[0])]
            back_pointer[(state, 0)] = None

        # calculating viterbi path log probability and back pointer for remaining states (time t = 1 to n-1)
        for t in range(1, len(obsv_sequence)):

            suffix_bi = obsv_sequence[t][-2:]
            suffix_tri = obsv_sequence[t][-3:]

            suffix_first, suffix_second = suffix_bi, suffix_tri
            if(suffix_bi in self.suffixes and suffix_tri in self.suffixes and self.suffixes[suffix_bi] < self.suffixes[suffix_tri]):
                suffix_first, suffix_second = suffix_tri, suffix_bi 

            for state in self.training_states:

                curr = float("-inf")
                back = None

                for prev_state in self.training_states:
                    alpha = 0
                    if (not ((state, obsv_sequence[t]) in self.emission_log_probs)):
                        if obsv_sequence[t] in self.training_obsvs or viterbi[(prev_state, t-1)] == float("-inf") or self.transition_log_probs[(prev_state, state)] == float("-inf")  or not(state in self.open_class_states):
                            alpha = float("-inf")
                        else:
                            if suffix_first in self.suffixes:
                                alpha = viterbi[(prev_state, t-1)] + self.transition_log_probs[(prev_state, state)] + self.suffix_emission_log_probs[(state, suffix_first)]
                            elif suffix_second in self.suffixes:
                                alpha = viterbi[(prev_state, t-1)] + self.transition_log_probs[(prev_state, state)] + self.suffix_emission_log_probs[(state, suffix_second)]
                            else:
                                alpha = viterbi[(prev_state, t-1)] + self.transition_log_probs[(prev_state, state)]
                    else:
                        if  viterbi[(prev_state, t-1)] == float("-inf") or self.emission_log_probs[(state, obsv_sequence[t])] == float("-inf") or self.transition_log_probs[(prev_state, state)] == float("-inf"):
                            alpha = float("-inf")
                        else:
                            alpha = viterbi[(prev_state, t-1)] + self.transition_log_probs[(prev_state, state)] + self.emission_log_probs[(state, obsv_sequence[t])]

                    if t == len(obsv_sequence)-1:
                        alpha += self.final_probs[state]

                    if alpha >= curr:
                        curr = alpha
                        back = prev_state

                viterbi[(state, t)] = curr
                back_pointer[(state, t)] = back

        state_sequence = []
        bestpathprob = float("-inf")
        bestpathstate = None

        # finding best viterbi path log probabilities of final time t = n-1
        for state in self.training_states:
            if viterbi[(state, len(obsv_sequence)-1)] >= bestpathprob:
                bestpathprob = viterbi[(state, len(obsv_sequence)-1)]
                bestpathstate = state

        state_sequence.append(bestpathstate)
        prev_state = back_pointer[(bestpathstate, len(obsv_sequence)-1)]

        # backpropagating through back pointers and inserting to state sequence
        for t in range(len(obsv_sequence)-2, -1, -1):
            state_sequence.insert(0, prev_state)
            prev_state = back_pointer[(prev_state, t)]

        return state_sequence

    def write_output(self, file, tagged_lines) -> None:
        """
            save tagging results to output file
        """
        with open(file, 'w', encoding='utf-8') as file:
            for tagged_line in tagged_lines:
                output_line = []
                # write as observation/state for each word
                for obsv, state in tagged_line:
                    output_line.append(f'{obsv}/{state}')
                file.write(' '.join(output_line))
                file.write('\n')

    # --------------------------------------------------------
    # --------------------------------------------------------
    # FOR COMPARISON
    # --------------------------------------------------------
    # --------------------------------------------------------

    def parse_tagged_input(self, lines) -> list:
        """
            parse tagged input file by splitting tokens into observations and states
        """
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

    def compare_result(self, file, tagged_pred_lines) -> None:
        """
            find accuracy of tagging done by HMM by comparing against pre-tagged ground truth file
        """
        lines = []
        with open(file, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                lines.append(line)

        parsed_test_lines = self.parse_tagged_input(lines)

        total, count = 0, 0

        seen, seen_correct_count, unseen, unseen_correct_count = 0, 0, 0, 0
        for i in range(len(parsed_test_lines)):
            for j in range(len(parsed_test_lines[i])):

                pred_obsv, pred_state = tagged_pred_lines[i][j]
                test_state, _ = parsed_test_lines[i][j]
                
                if pred_obsv in self.training_obsvs:
                    seen += 1
                else:
                    unseen += 1

                if (pred_state == test_state):
                    count += 1
                    if pred_obsv in self.training_obsvs:
                        seen_correct_count += 1
                    else:
                        unseen_correct_count += 1

                total += 1

        print(f'Accuracy: {100*count/total}')
        print(f'Total Seen: {seen}')
        print(f'Seen Accuracy: {100*seen_correct_count/seen}')
        print(f'Total Unseen: {unseen}')
        print(f'Unseen Accuracy: {100*unseen_correct_count/unseen}')

def main():

    hm = HmmModel()
    hm.load_weights('hmmmodel.txt')

    try:
        input_file = sys.argv[1]
    except:
        input_file = "data/it_isdt_dev_raw.txt"

    lines = []
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            lines.append(line)

    parsed_lines = hm.parse_input(lines)
    tagged_lines = hm.run(parsed_lines)
    output_file = "hmmoutput.txt"
    hm.write_output(output_file, tagged_lines)

    hm.compare_result('data/it_isdt_dev_tagged.txt', tagged_lines)

if __name__ == "__main__":
    main()