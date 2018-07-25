This directory contains datasets and implementations of baseline models and our methods used in our papers [Filling Missing Paths: Modeling Co-occurrences of Word Pairs and Dependency Paths for Recognizing Lexical Semantic Relations][1].  
We used python 3.6.1.  
These codes require chainer==2.1.0, sklearn, numpy,  spacy==1.9.0, and h5py.  
This software includes the work that is distributed in the Apache License 2.0.  

--------------------------------------------------------------------------------------------------------------------
# Usage
Please follow the process below.

1. Prepare GloVe. Please download 50-d GloVe from https://nlp.stanford.edu/projects/glove/, and put glove.6B.50d.txt into /work and run glove_process.py
	$ python glove_process.py -g glove.6B.50d.txt

2. Create (w1, w2, path) triples from your own corpus.
Put your own corpus, such as wikipedia corpus, into /corpus and run;
	$ cd corpus
	$ ./create_triples.sh <your corpus>

3. Create unsupervised learning data.
	$ ./unsp_data_creating.sh

4. Run the unsupervised learning of our model of P(path|w1, w2).
	$ python unsp_model_training.py -d unsp_data.dump -o unsp_model

5. Process the datasets for supervised learning.
	$ python datasets_process.py

6. Augment path data with the model of P(path|w1, w2). For example;
	$ python unsp_data_augment.py -d datasets -u unsp_model/unsp_model.model -k 1 -o datasets_aug1

7. Please copy relations.txt of each dataset into augmented one.
	$ cp datasets/BLESS/relations.txt datasets_aug1/BLESS/

8. Run the supervised learning. For example;
	NPB
	$ python supervised_path_based.py --data_prefix datasets/BLESS -o result

	NPB
	$ python supervised_path_based.py --data_prefix datasets_aug1/BLESS -o result

	LexNET
	$ python supervised_lexnet.py --data_prefix datasets/BLESS -o result

	LexNET_h
	$ python supervised_lexnet.py --data_prefix datasets_aug1/BLESS -lh 1 -o result/BLESS

	LexNET+Aug
	$ python supervised_lexnet.py --data_prefix datasets_aug1/BLESS -o result

	LexNET+Rep
	$ python supervised_lexnet_rep.py --data_prefix datasets/BLESS -u unsp_model/unsp_model.model -o result

	LexNET+Aug+Rep
	$ $ python supervised_lexnet_rep.py --data_prefix datasets_aug1/BLESS -u unsp_model/unsp_model.model -o result


[1]:http://aclweb.org/anthology/N18-1102
