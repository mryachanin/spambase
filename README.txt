Writeup:
The main writeup for this project is called spam_or_not_spam.pdf

Data:
The testing and training data used is compressed in data.zip.
Feature selected data is prefixed spambase-fs.
Non-feature selected data is prefixed spambase.
For both datasets, there are 4 files. 
spambase(-fs).data is the vanilla data.
spambase(-fs)-random.data is shuffled vanilla data for training (95% of the data).
spambase(-fs)-validation.data is shuffled vanilla data used for final validation (5% of the data).

Code:
The main method is in neural_network.runners.EmailSpamClassifier
Usage: java neural_network.runners.EmailSpamClassifier train training_data_filepath validation_data_filepath
       java neural_network.runners.EmailSpamClassifier test validation_data_filepath neural_network_save_filepath

