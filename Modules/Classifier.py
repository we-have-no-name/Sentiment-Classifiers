import tensorflow as tf
import numpy as np
from TweetToWordIndices import TweetToWordIndices
from AccuracyAnalysis import AccuracyAnalysis, TrainStats
from NNGraph import NNGraph
from DataSetReader import DataSetReader
import os, time
import json

class Classifier:
	'''
	Classifier used for classifying tweets using TensorFlow
	'''
	def __init__(self, graph=None, data_set=None, user_config_filename='config.json', restore_saved_session=True, **kwargs):
		'''
		Sets and initializes the classifier's TensorFlow graph and session, the data_set and the accuracy analyzer
		args:
		graph: object of NNGraph
		data_set: object of DataSetReader
		user_config_filename: name of the file that has the user configurations
		restore_saved_session: restore a saved session instead of training one (can still be trained after restore)
		'''
		try:
			json_config_file = open(user_config_filename)
		except FileNotFoundError:
			json_config_file = self.set_config_file(user_config_filename)
			
		user_config = json.load(json_config_file)
		self.data_path = user_config.get('data_path', os.getcwd())
		self.checkpoint_name = user_config.get('checkpoint_name', os.path.join('DefaultSession', 'default'))
		
		if graph != None: self.graph = graph
		else: self.graph = NNGraph(use_default_network = True)
		if data_set==None: data_set = DataSetReader()
		self.data_set = data_set
		self.num_steps = self.graph.num_steps
		self.classes = self.graph.classes
		self.sess = tf.Session(graph=self.graph.graph)
		self.tweet_to_indices = TweetToWordIndices(assumed_max_length=self.num_steps)
		self.train_stats = TrainStats()
		self.accuracy_analysis = AccuracyAnalysis(train_stats=self.train_stats, classes = self.classes, data_set = self.data_set)
		
		if restore_saved_session: self.restore_session()
		else: self.init_session()
		self.session_init_time = 'UTC'+time.strftime("%y%m%d-%H%M%S", time.gmtime())
		
		if kwargs.get('set_note', False)==True: self.accuracy_analysis.set_note();
		
	def set_config_file(self, user_config_filename):
		'''
		Set a user config file with the path that has the Embedding, Sessions folders
		args:
		user_config_filename: the file name of the json config file the has the data_path
		'''
		user_config = {}
		user_config['data_path'] = input("Enter the path to the folder that has the Embedding, Sessions folders:\n")
		user_config['checkpoint_name'] = input("Enter the path to the session checkpoint relative to data_path/Sessions:\n")

		json_config_file = open(user_config_filename, 'w+')
		json.dump(user_config, json_config_file)
		json_config_file.seek(0)
		return json_config_file

	def init_session(self):
		'''
		Initializes a new TensorFlow session
		'''
		self.sess.run(self.graph.global_variables_initializer)
		self.graph.embedding_saver.restore(self.sess, os.path.join(self.data_path, "d"+ str(self.graph.embedding_dim) + "_word_embedding", "TF_Variables", "Embedding"))
		return True
		
	def train(self, iters=100, inputs=None, targets=None, batch_size=50, max_train=None, max_test=None, **kwargs):
		'''
		Trains the classifier (can be called multiple times and after a session is restored)
		args:
		inputs: a list of strings to classify
		targets: a numpy array of shape (inputs_length, classes_count) having each string's probabilitis for each class
		iters: training iterations
		batch_size: the batch size for training
		max_train: count of items from inputs used for training
		max_test: count of items from inputs used for testing
		'''
		if(inputs==None and targets==None):
			inputs = self.data_set.tweets
			targets = self.data_set.sents_sc_np
		inputs_count = len(inputs)
		if max_train is None: max_train=int(inputs_count*0.8)
		if max_test is None: max_test=int(inputs_count*0.2)
		drop_out = kwargs.get('drop_out', 0.0)
		print_stats = kwargs.get('print_stats', False)
		checkpoint_distance = kwargs.get('checkpoint_distance', 5)
		
		iter_train_probs = np.zeros((max_train, self.classes))
		
		inputs_keys = np.zeros((len(inputs), self.tweet_to_indices.assumed_max_length))
		for i in range(len(inputs)):
			inputs_keys[i] = self.tweet_to_indices.tweet_to_word_indices(inputs[i])
		
		train_time0 = self.train_stats.train_time
		start_time = time.time();
		for i in range(iters):
			checkpoint = None
			if (i+1)%checkpoint_distance == 0: checkpoint = (i+1)//checkpoint_distance 
			if i==0: checkpoint=0

			# Train the model and get train results
			for train in range(int(max_train/batch_size)):
				batch_start = train*batch_size; batch_end = (train+1)*batch_size
				if (train+2)*batch_size>max_train: batch_end=max_train
##				np_inputs = inputs_embedded[batch_start:batch_end, :self.graph.num_steps]
				np_inputs_keys = inputs_keys[batch_start:batch_end, :self.graph.num_steps]
				np_targets = targets[batch_start:batch_end]
				if checkpoint is not None:
##					_, np_probs = self.sess.run([self.graph.opt_op, self.graph.probs], feed_dict={self.graph.inputs: np_inputs, self.graph.targets_mc: np_targets, self.graph.drop_out:drop_out})
					_, np_probs = self.sess.run([self.graph.opt_op, self.graph.probs], feed_dict={self.graph.inputs_keys: np_inputs_keys, self.graph.targets_mc: np_targets, self.graph.drop_out:drop_out})
					iter_train_probs[batch_start:batch_end] = np_probs
				else:
##					_ = self.sess.run([self.graph.opt_op], feed_dict={self.graph.inputs: np_inputs, self.graph.targets_mc: np_targets})
					_ = self.sess.run([self.graph.opt_op], feed_dict={self.graph.inputs_keys: np_inputs_keys, self.graph.targets_mc: np_targets})
			self.train_stats.train_iters += 1
			
			if checkpoint is None: continue

			# Get test results
##			np_test_inputs = inputs_embedded[max_train:: max_train+max_test, :self.graph.num_steps]
			np_test_inputs_keys = inputs_keys[max_train: max_train+max_test, :self.graph.num_steps]
			np_test_targets = targets[max_train: max_train+max_test]
##			np_test_probs, = sess.run([self.graph.probs], feed_dict={self.graph.inputs:  np_test_inputs})
			iter_test_probs, = self.sess.run([self.graph.probs], feed_dict={self.graph.inputs_keys:  np_test_inputs_keys})

			end_time = time.time()
			eta = (end_time - start_time)*(iters-i-1)/(i+1)
			self.train_stats.train_time = train_time0 + end_time-start_time
			self.test_acc, self.max_test_acc = self.accuracy_analysis.add_probs(iter_train_probs, iter_test_probs)
			if print_stats: print('{}\nETA {}\n'.format(self.accuracy_analysis.statistics, self.accuracy_analysis.sec2clock(eta)))
			
		return self.test_acc, self.max_test_acc
			

	def save_session(self):
		'''
		Saves the trained state of the classifier
		'''
		save_path = os.path.join(self.data_path, 'Sessions', self.checkpoint_name)
		if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
		self.graph.train_saver.save(self.sess, save_path)
		return True
	
	def restore_session(self):
		'''
		Restores a trained state of the classifier
		'''
		self.graph.train_saver.restore(self.sess, os.path.join(self.data_path, 'Sessions', self.checkpoint_name))
		return True
	
	def predict(self, inputs):
		'''
		Predicts the class probabilities for each input
		inputs: list of strings to classify
		'''
		inputs_keys = np.zeros((len(inputs), self.tweet_to_indices.assumed_max_length))
		for i in range(len(inputs)):
			inputs_keys[i] = self.tweet_to_indices.tweet_to_word_indices(inputs[i])
##		np_inputs = inputs[:, :self.graph.num_steps]
		np_inputs_keys = inputs_keys[:, :self.graph.num_steps]
##		output_probs, = self.sess.run([self.graph.probs], feed_dict={self.graph.inputs: inputs})
		output_probs, = self.sess.run([self.graph.probs], feed_dict={self.graph.inputs_keys: np_inputs_keys})
		return output_probs

def main():
	'''
	Train the classifier if the file is run as a script
	'''
	print("Training\n")
	c=Classifier(restore_saved_session=False)
	c.train(100, print_stats=True)
	save = input("Save session? (y, n) [y]:")
	if save == '' or save == 'y':
		c.save_session()
		print("Session saved")
	print("Done")

if __name__ == '__main__': main()
