<a name="readme-top"></a>
# Hidden Markov Model - Part of Speech Tagging

## Description

A part of speech tagging tool developed using hidden markov model. It uses tagged training data to learn weights for the tranisiton and emission matrices of a hidden markov model. 
The model can then be run on untagged input text and will output a file tagging the part-of-speech for 
each word in the input file. It also offers functionality to compare accuracy against ground truth tagged results. 
The model achieves a performace of 87.98% for Italian ISDT and 85.92% for Japanese GSD Datasets. 

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


## Authors

- Dhananjai Chand

## Acknowledgments

* [Jurafsky and Martin, chapter 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf)
* [Jurafsky and Martin, appendix A](https://web.stanford.edu/~jurafsky/slp3/A.pdf)

<p align="right">(<a href="#readme-top">back to top</a>)</p>