<a name="readme-top"></a>
# Hidden Markov Model - Part of Speech Tagger

## Description

A part-of-speech tagging tool developed written in Python that uses Hidden Markov Model. It uses tagged training data to learn probability scores for the transition and emission matrices of the Hidden Markov Model. The model can then be run on an untagged input text and will output a file tagging the part-of-speech for 
each word in the input file. It also offers functionality to compare accuracy against ground truth tagged results. 
The model achieves an accuracy of 94.35% for Italian ISDT and 91.91% for Japanese GSD Datasets which lie in the data folder. 

## Getting Started

### Dependencies

* Python3
* NumPy

### Installing

Use the following steps for installation.

1. Clone the repo
   ```sh
   git clone https://github.com/dhananjaichand98/Hidden-Markov-POS-Tagger.git
   ```
3. Install required Python packages
   ```sh
   pip3 install -r requirements.txt
   ```

### Executing program

There are two programs: hmmlearn.py will learn a hidden Markov model from the training data, and hmmdecode.py will use the model to tag new data.

* The learning program will be invoked in the following way. It will generate a file name hmmmodel.txt.
    ```
    python hmmlearn.py /path/to/input
    ```
* The tagging program will be invoked in the following way. It will generate a file named hmmoutput.txt.
    ```
    python hmmdecode.py /path/to/input
    ```

Alternatively, 
* execute bash script to run both the files and get tagging result.
    ```
    bash bash.sh /path/to/training_input /path/to/raw_input
    ```

## Data

The data folder contains *_train_tagged.txt, *_dev_tagged.txt and *_dev_raw.txt files in the following format:

* *_train_tagged.txt: A file with tagged training data in the word/TAG format, with words separated by spaces and each sentence on a new line.
* *_dev_raw.txt: A file with untagged development data, with words separated by spaces and each sentence on a new line.
* *_dev_tagged.txt: A file with tagged development data in the word/TAG format, with words separated by spaces and each sentence on a new line, to serve as an answer key.

## Authors

- Dhananjai Chand

## Acknowledgment

* [Jurafsky and Martin, chapter 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf)
* [Jurafsky and Martin, appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>