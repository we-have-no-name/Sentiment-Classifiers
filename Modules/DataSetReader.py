import csv, json
import numpy as np
import os
import random

class DataSetReader():
	'''
	A reader for the dataset's csv files, it extracts tweet texts and sentiments
	attributes:
	self.tweets: has the tweet texts
	self.sents_np: has the first sentiment class of each tweet
	self.sents_sc_np: has the probability of each class for each tweet
	usage:
	either set use_default_folder=True when creating the object or manually use read_file() then lists_to_arrays()
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
		self.lists_to_arrays(shuffle_seed)
		return True

	def lists_to_arrays(self, shuffle_seed=0):
		'''
		Create numpy arrays from the sentiment lists
		args:
		shuffle_seed: the seed to be used to shuffle the tweets, Use None to skip shuffling
		'''
		if shuffle_seed is not None: self.shuffle_tweets(shuffle_seed)
		
		self.sents_np = np.array(self.sentiments, np.int16)
		
##		self.sents_mh_np = np.zeros((self.sents_np.shape[0], self.classes), np.bool) #mh: multihot
##		for i in range(self.sents_np.shape[0]):
##			self.sents_mh_np[i, self.sentiments_lists[i]] = 1
		
		priority_probs=[np.array([1]), np.array([2/3, 1/3]), np.array([3/6, 2/6, 1/6])]		
		self.sents_sc_np = np.zeros((self.sents_np.shape[0], self.classes), np.float32) #sc: soft classes
		for i in range(self.sents_np.shape[0]):
			sc_indices = self.sentiments_lists[i]
			self.sents_sc_np[i, sc_indices] = priority_probs[len(sc_indices)-1]
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
		create_arrays: create the numpy arrays for the sentiment lists, set False to read more files first
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
		if create_arrays: self.lists_to_arrays(shuffle_seed)
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

