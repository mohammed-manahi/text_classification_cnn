from __future__ import unicode_literals
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from numpy import array
import keras.preprocessing.text, keras.preprocessing.sequence

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents
 
# load all training reviews
positive_docs = process_docs(r'C:\Users\moham\Lab\text_classification_cnn\data\txt_sentoken\pos', vocab, True)
negative_docs = process_docs(r'C:\Users\moham\Lab\text_classification_cnn\data\txt_sentoken\neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = keras.preprocessing.text.Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)


# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs(r'C:\Users\moham\Lab\text_classification_cnn\data\txt_sentoken\pos', vocab, False)
negative_docs = process_docs(r'C:\Users\moham\Lab\text_classification_cnn\data\txt_sentoken\neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1