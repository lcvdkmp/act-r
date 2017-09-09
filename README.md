
## Usage
The model has two modes: run and train. The mode of the model is specified by
the first command line argument.
In run-mode, the model will run on a single test sentence.
In train-mode, the model will train itself to fit the given data.
Help for both modes can be found bu executing: `python model run -h` and
`python model train -h`.

To train the basic reading model, run:
`python model train data/fillers.txt data/results_fillers_RTs.csv`

To get the results of the pre-trained basic reading model, run:
`python model train data/fillers.txt data/results_fillers_RTs.csv -r data/final_results_basic.csv`

To train the advanced model, run:
`model train data/target_sentences.txt data/pronouns_RTs.csv -n data/final_results_basic.csv`

To get the results of the pre-trained advanced model, run:
`python model train data/target_sentences.txt data/pronouns_RTs.csv -n data/final_results_basic.csv -r final_results_noun_full.csv`

To train the advanced model on 'mis match' sentences only, run:
`model train data/target_sentences.txt data/pronouns_RTs.csv -n data/final_results_basic.csv` -f allow-mis-match

To get the results of the pre-trained advanced model on 'mis match' sentences only, run:
`python model train data/target_sentences.txt data/pronouns_RTs.csv -n data/final_results_basic.csv -r final_results_noun_mis_match.csv` -f allow-mis-match
