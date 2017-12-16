This directory contains datasets and implementations of baseline models and our methods used in our papers.
We used python 3.6.1.
These codes require chainer==2.1.0, sklearn, numpy,  spacy==1.9.0, and h5py.

--------------------------------------------------------------------------------------------------------------------
Usage
Please follow the process below.

1. Create (w1, w2, path) triples from your own corpus.
Put your own corpus, such as wikipedia corpus, into /corpus and run;
	$ cd /corpus
	$ ./create_triples.sh <your corpus>

2. Create unsupervised learning data.
	$ ./unsp_data_creating.sh
	
3. Run the unsupervised learning of our model of P(path|w1, w2).
	$ python unsp_model_training.py -d unsp_data.dump -o unsp_model

4. Process the datasets for supervised learning.
	$ python datasets_process.py
	
5. Augment path data with the model of P(path|w1, w2).
	$ 



