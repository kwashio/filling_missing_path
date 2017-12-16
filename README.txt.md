This directory contains implementations of baseline models and our methods.
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
	




