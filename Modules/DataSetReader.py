from TweetToWordIndices import TweetToWordIndices
import csv, json
import numpy as np
import os
import random

class DataSetReader():
	'''
	A reader for the dataset's csv files, it extracts tweet texts and sentiments
	then indexes words in tweets and creates target arrays from the labels of the tweets
	attributes:
	self.tweets: has the tweet texts
	self.tweets_indices: has the embedding index of each word in the tweets
	self.sents_np: has the first sentiment class of each tweet
	self.sents_sc_np: has the probability of each class for each tweet
	usage:
	either set use_default_folder=True when creating the object or manually use read_file() then sentiment_lists_to_arrays()
	'''
	def __init__(self, use_default_folder=True, classes=8, user_config_filename='config.json'):
		'''
		Read the data_path of the folder that has the DataSet folder
		args:
		use_default_folder: Read all files in the DataSet folder
		classes: the count of the classes used in the files
		user_config_filename: the file name of the json config file the has the data_path
		'''			
		try:
			json_config_file = open(user_config_filename)
		except FileNotFoundError:
			json_config_file = self.set_config_file(user_config_filename)
			
		user_config = json.load(json_config_file)
		self.data_path = user_config.get('data_path', os.getcwd())
		self.classes = classes
		self.sentiments = []
		self.sentiments_lists = []
		self.tweets = []
		self.text_indexer = TweetToWordIndices()
		
		if use_default_folder: self.read_all_files(os.path.join(self.data_path, 'DataSet'))
		
	def set_config_file(self, user_config_filename):
		'''
		Set a user config file with the path that has the DataSet folder
		args:
		user_config_filename: the file name of the json config file the has the data_path
		'''
		user_config = {}
		user_config['data_path'] = input("Enter the path to the folder that has the DataSet folder:\n")

		json_config_file = open(user_config_filename, 'w+')
		json.dump(user_config, json_config_file)
		json_config_file.seek(0)
		return json_config_file

	def read_all_files(self, folder_path, shuffle_seed=0):
		'''
		Read all files in the DataSet folder
		args:
		folder_path: the folder that has the data set files
		shuffle_seed: the seed to be used to shuffle the tweets, Use None to skip shuffling
		'''
		dir_contents = os.listdir(folder_path)
		for name in dir_contents:
			if name[-4:]=='.csv': self.read_file(os.path.join(folder_path, name), create_arrays=False)
		self.sentiment_lists_to_arrays(shuffle_seed)
		self.tweets_to_indices()
		return True

	def sentiment_lists_to_arrays(self, shuffle_seed=0):
		'''
		Create numpy arrays from the sentiment lists
			also updates the classes statistics
		args:
		shuffle_seed: the seed to be used to shuffle the tweets, Use None to skip shuffling
		'''
		if shuffle_seed is not None: self.shuffle_tweets(shuffle_seed)

		self.size = len(self.sentiments)
		self.sents_np = np.array(self.sentiments, np.int16)
		
		priority_probs=[np.array([1]), np.array([2/3, 1/3]), np.array([3/6, 2/6, 1/6])]		
		self.sents_sc_np = np.zeros((self.sents_np.shape[0], self.classes), np.float32) #sc: soft classes
		for i in range(self.sents_np.shape[0]):
			sc_indices = self.sentiments_lists[i]
			self.sents_sc_np[i, sc_indices] = priority_probs[len(sc_indices)-1]

		self.sents_mh_np = self.sents_sc_np > 0 #mh: multihot

		self.multiclass_count = np.sum(np.sum(self.sents_mh_np, 1)>1)
		self.multiclass_ratio = self.multiclass_count/self.size
		self.class_counts = np.bincount(self.sents_np)
		return True
		
	def shuffle_tweets(self, shuffle_seed):
		'''
		Shuffle the tweets and their sentiments in lists
		args:
		shuffle_seed: the seed to be used to shuffle the tweets, Use None to skip shuffling
		'''
		random.seed(shuffle_seed)
		indices_shuffled = list(range(len(self.tweets)))
		random.shuffle(indices_shuffled)
		self.tweets = [self.tweets[i] for i in indices_shuffled]
		self.sentiments = [self.sentiments[i] for i in indices_shuffled]
		self.sentiments_lists = [self.sentiments_lists[i] for i in indices_shuffled]
		return True
			
	def read_file(self, file_path, encoding='utf-8-sig', shuffle_seed=0, create_arrays=True):
		'''
		Read a csv file and extract tweets and sentiments
		args:
		file_path: the path of the csv file
		encoding: the encoding of the csv file
		shuffle_seed: the seed to be used to shuffle the tweets, Use None to skip shuffling
		create_arrays: create the numpy arrays to be used for training, set False to read more files first
		'''
		tweets_file = open(file_path, 'r', encoding=encoding)
		tweets_csv = csv.reader(tweets_file)
		for line in tweets_csv:
			if len(line)<3 or line[1]=='' : continue
			sents = self._multiclass_sentiment(line[2:])
			if len(sents)==0: continue
			sent = sents[0]
			self.sentiments.append(sents[0])
			self.sentiments_lists.append(sents)
			self.tweets.append(line[1])
		if create_arrays:
			self.sentiment_lists_to_arrays(shuffle_seed)
			self.tweets_to_indices()
		return True
		

	def _multiclass_sentiment(self, sent_cells):
		'''
		[Internal] Reads the sentiment indices from the csv rows
		args:
		sent_cells: a list of the values of the cells that have the indices
		'''
		sentiment_list = []
		for i in sent_cells:
			if i == '' or not i.isdigit() : continue
			sentiment_list.append(int(i) - 1)
		return sentiment_list

	def tweets_to_indices(self):
		'''converts the tweets list to a list of arrays of tweets indices
			also updates the matching statistics
		'''
		self.size = len(self.tweets)
		self.tweets_indices = self.text_indexer.tweets_to_word_indices(self.tweets)
		self.max_length = self.text_indexer.get_max_length(self.tweets_indices)
		match_stats = self.text_indexer.get_match_statistics(self.tweets, self.tweets_indices)
		self.match_ratio = match_stats['match_ratio']
		self.unmatched_words_count = match_stats['unmatched_words_count']
		self.unmatched_words_counts = match_stats['unmatched_words_counts']
		return self.tweets_indices
		
