import preprocess_twitter
import html
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle, csv, json
import os
import numpy as np
import re

class TweetToWordIndices():
	'''
	Converts the text of tweets into the word index of each word from the GloVe word embedding
	The text is preprocessed to be the best match possible to the vocabulary in GloVe
		First the text is preprocessed as recommended by GloVe (this should be the same preprocessing applied to the GloVe corpus)
		then html escape codes are replaced by their equivalent characters
		And punctuations are split from words
		And emoji are split from each other and replaced with special tags available for emotions in the vocabulary
	After preprocessing each word is replaced with its word index from the embedding dictionary
	'''
	def __init__(self, embedding_dim=200, vocab_size=int(1.2e6), assumed_max_length=90, user_config_filename='config.json'):
		'''Read the embedding dictionaries'''
		try:
			json_config_file = open(user_config_filename)
		except FileNotFoundError:
			json_config_file = self.set_config_file(user_config_filename)
			
		user_config = json.load(json_config_file)
		self.data_path = user_config.get('data_path', os.getcwd())
		word_dict2_file = open(os.path.join(self.data_path, "d" + str(embedding_dim) +"_word_embedding", "word_dict2.pickle"), 'rb')
		self.word_dict2 = pickle.load(word_dict2_file)
		word_dict2_file.close()
		
		self.tokenizer = TreebankWordTokenizer()
		self.embedding_dim = embedding_dim
		self.vocab_size = vocab_size
		self.assumed_max_length = assumed_max_length

	def set_config_file(self, user_config_filename):
		'''Set a user config file with the path that has the embedding dictionaries'''
		user_config = {}
		user_config['data_path'] = input("Enter the path to the folder that has embedding dictionaries:\n")

		json_config_file = open(user_config_filename, 'w+')
		json.dump(user_config, json_config_file)
		json_config_file.seek(0)
		return json_config_file
		
	def unescape_html(self, txt):
		'''Replaces HTML escape codes with their equivalent characteres'''
		return html.unescape(txt)

	def split_punctuations(self, txt):
		'''Splits punctuations from words'''
		txt = re.sub(r"([^\w ']|([^a-zA-Z]'|'[^a-zA-Z ]|^'|'$))", r' \1 ', txt, flags=re.M)
		return txt

	def split_emojis(self, txt):
		'''Splits emoji from each other'''
		emojis='[\U0001F601-\U0001F64F\U00002702-\U000027B0\U0001F680-\U0001F6C0\U000024C2-\U0001F251\U0001F600-\U0001F636\U0001F681-\U0001F6C5\U0001F30D-\U0001F567]'
		return re.sub(r" *({}) *".format(emojis), r' \1 ', txt)

	def replace_emojis(self, txt):
		'''Replaces emoji with the special tags available for emotions in the vocabulary'''
		txt=re.sub('[\U0000FE00-\U0000FE0F]', '', txt) #remove variation selectors
		txt=re.sub('[\U0001F3FB-\U0001F3FF]', '', txt) #remove color tones
		smile_ug = '[\U0001F603\U0001F604\U0001F600]'
		lolface_ug = '[\U0001F602\U0001F606]'
		sadface_ug = '[\U0001F614\U0001F616\U0001F622\U0001F625\U0001F629\U0001F62D\U0001F630\U0001F63F]'
		neutralface_ug = '[\U0001F610]'
		heart_ug = '[\U0001F60D\U0001F618\U0001F63B\U00002764\U0001F491-\U0001F49F]'
		txt = re.sub(smile_ug, " ᐸsmileᐳ ", txt)
		txt = re.sub(lolface_ug, " ᐸlolfaceᐳ ", txt)
		txt = re.sub(sadface_ug, " ᐸsadfaceᐳ ", txt)
		txt = re.sub(neutralface_ug, " ᐸneutralfaceᐳ ", txt)
		txt = re.sub(heart_ug, " ᐸheartᐳ ", txt)
		return txt

	def tokenize(self, txt):
		'''Preprocess text and tokenize it'''
		txt = preprocess_twitter.tokenize(txt)	
		txt = self.unescape_html(txt)
		txt = self.split_punctuations(txt)
		txt = self.split_emojis(txt)
		txt = self.replace_emojis(txt)
		words = self.tokenizer.tokenize(txt)
		return words

	def words_to_indices(self, words):
		'''Gets the index of each word from the embedding dictionary'''
		word_indices=np.array([self.vocab_size-2 for _ in range(self.assumed_max_length)])
		for i in range(len(words)) :
			word_indices[i] = self.word_dict2.get(words[i], self.vocab_size-1)
		return word_indices

	def tweet_to_word_indices(self, txt):
		'''Returns a list of the word indices of each word in the tweet'''
		words = self.tokenize(txt)
		return self.words_to_indices(words)
		
	
