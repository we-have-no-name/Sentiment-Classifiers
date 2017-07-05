import numpy as np
import matplotlib.pyplot as plt
import json, os, shutil

class AccuracyAnalysis:
	'''
	Analyze accuracy of the classifier over time
	'''
	def __init__(self, classes=8, data_set=None, log_folder='Temp', graph_description=None):
		'''
		Initialize accuracy arrays
		args:
		classes: count of the classes used by the classifier
		data_set: the data_set used to train the classifier (can be replaced later)
		log_folder: the folder to save the logs to
		graph_description: the description of the neural network graph
		'''
		self.classes = classes
		self.data_set = data_set
		self.log_folder = log_folder
		self.graph_description = graph_description
		self.train_accs = np.zeros(0)
		self.test_accs = np.zeros(0)
		self.test_accs2 = np.zeros(0)
		self.test_accs3 = np.zeros(0)
		self.max_train_acc, self.max_test_acc, self.max_test_acc2, self.max_test_acc3 = 0, 0, 0, 0
		self.train_iters = 0
		self.saved_iters = []
		self.train_time = 0
		self.sess_save_path = None
		self.script_path = None
		self.note = ''

	def add_probs(self, train_probs, test_probs, data_set=None):
		'''
		Receive output probabilities for the data_set from the classifier
		args:
		train_probs: train set probabilities
		test_probs: test set probabilities
		data_set: the data_set used to train the classifier 
		'''
		self.train_probs = train_probs
		self.test_probs = test_probs
		self.max_train = len(self.train_probs)
		self.max_test = len(self.test_probs)
		if data_set is not None:
			self.data_set = data_set
		if self.train_iters == 1 or data_set is not None:
			self.update_data_set_statistics()
		self.train_baseline, self.test_baseline = self.baseline_pf(self.max_train, self.max_test)
		self.train_targets = self.data_set.sents_sc_np[:self.max_train]
		self.test_targets = self.data_set.sents_sc_np[self.max_train:self.max_train+self.max_test]

		self.saved_iters.append(self.train_iters)
		self.update_accuracies()
		self.update_statistics()
		self.log()
		return self.test_acc, self.max_test_acc

	def update_accuracies(self):
		'''
		Calculate and update all accuracies for the data set
		'''
		self.trains_acc0, self.train_acc0 = self.acc0(self.train_probs, self.train_targets)
		trains_acc_r, self.trains_acc, self.train_acc = self.acc(self.train_probs, self.train_targets)
		self.max_train_acc = max(self.train_acc, self.max_train_acc)
		self.train_accs = np.concatenate([self.train_accs, np.array([self.train_acc])])
		
		self.tests_acc0, self.test_acc0 = self.acc0(self.test_probs, self.test_targets)
		tests_acc_r, self.tests_acc, self.test_acc = self.acc(self.test_probs, self.test_targets)
		self.max_test_acc = max(self.test_acc, self.max_test_acc)
		self.test_accs = np.concatenate([self.test_accs, np.array([self.test_acc])])
		self.tests_acc2, self.test_acc2 = self.acc2(self.tests_acc0, self.tests_acc)
		self.max_test_acc2 = max(self.test_acc2, self.max_test_acc2)
		self.test_accs2 = np.concatenate([self.test_accs2, np.array([self.test_acc2])])
		self.tests_acc3, self.test_acc3 = self.acc3(self.test_probs, self.test_targets, self.tests_acc0, tests_acc_r)
		self.max_test_acc3 = max(self.test_acc3, self.max_test_acc3)
		self.test_accs3 = np.concatenate([self.test_accs3, np.array([self.test_acc3])])
		return True
		
	def acc0(self, probs, targets):
		'''
		Checks if the top output class matches the top target class
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		'''
		accs0 = np.argsort(probs)[:,-1] == np.argsort(targets)[:,-1]
		acc0 = np.mean(accs0)
		return accs0, acc0

	def acc(self, probs, targets):
		'''
		Calculates min(output_prob, target_prob) for each class and each item
		This means how much of the required probability has been reached
		args:
		probs: output probabilities from the classifier
		targets: target probabilities
		return:
		accs_r: accuracies per example per class
		accs: sum of accuracies per example
		acc: mean of accuracies of all examples
		'''
		accs_r = np.minimum(probs, targets);
		accs = np.sum(accs_r, 1)
		acc = np.mean(accs)
		return accs_r, accs, acc

	def acc2(self, accs0, accs):
		'''
		Calculates max(acc0, acc) for each item
		This means if the top class is matched, give the full score, else give acc
		args:
		accs0: accs0 from method acc0
		accs: accs from method acc
		return:
		accs2: sum of accuracies per example
		acc2: mean of accuracies of all examples
		'''
		accs2 = np.maximum(accs0, accs)
		acc2 = np.mean(accs2)
		return accs2, acc2

	def acc3(self, probs, targets, accs0, accs_r):
		'''
		Give the full class score if prob reaches 50% of target
		then calculates max(acc3, acc) for each item
		then calculates max(acc3, acc0) for each item
		this means if for any class half the required probability is reached, the full class score is given
		otherwise acc is given
		and if the top prob matches the top target, the full item score is given
		args:
		accs0: accs0 from method acc0
		accs_r: accs_r from method acc
		return:
		accs3: sum of accuracies per example
		acc3: mean of accuracies of all examples
		'''
		accs3 = np.sum(np.maximum(accs_r, np.minimum((probs> 0.5 * targets), targets)), 1)
		accs3 = np.maximum(accs3, accs0)
		acc3 = np.mean(accs3)
		return accs3, acc3		
	
	def baseline_pf(self, max_train, max_test):
		'''
		Calculate baselines
		'''
		train_targets = self.data_set.sents_sc_np[:max_train]
		test_targets = self.data_set.sents_sc_np[max_train:max_train+max_test]
		
		top_probs = np.zeros((3, self.classes))
		priority_probs = [np.array([1]), np.array([2/3, 1/3]), np.array([3/6, 2/6, 1/6])]
		def baseline():
			baseline = 0
			for i in range(3): 
				baseline = max(baseline, np.mean(np.sum(np.minimum(selected_sents, top_probs[i]), 1)))
##			baseline = np.mean(np.sum(np.minimum(selected_sents, top=_probs[np.sum(selected_sents>0, 1)-1]), 1))
			return baseline
		
		selected_sents = train_targets
		top_indices = np.argsort(np.sum(selected_sents, 0))[::-1][:3]
		for i in range(3): top_probs[i, top_indices[:i+1]] = priority_probs[i]
		train_baseline = baseline()
		selected_sents = test_targets
		test_baseline = baseline() if max_test!=0 else 0
		return train_baseline, test_baseline
		
	def update_statistics(self):
		'''
		Update the statistics of accuracy and training
		'''
		iters = 'iterations ({}:{}):\n'.format(self.saved_iters[0], self.saved_iters[-1])
		latest_acc = 'train_acc: {:.3}  \ttest_acc: {:.3}\n'.format(self.train_accs[-1], self.test_accs[-1])
		top_acc = 'top train_acc: {:.3}\ttop test_acc: {:.3} @{}\n'.format(max(self.train_accs), max(self.test_accs), self.saved_iters[np.argmax(self.test_accs)])
		baseline = 'train_baseline: {:.3}\ttest_baseline: {:.3}\n'.format(self.train_baseline, self.test_baseline)
		pass_test = 'passing test examples: {:.3}\n'.format(np.mean(self.tests_acc>0.5))
		test_acc2 = 'test_acc2: {:.3}  \ttop test_acc2: {:.3} @{}\n'.format(self.test_accs2[-1], max(self.test_accs2), self.saved_iters[np.argmax(self.test_accs2)])
		test_acc3 = 'test_acc3: {:.3}  \ttop test_acc3: {:.3} @{}\n'.format(self.test_accs3[-1], max(self.test_accs3), self.saved_iters[np.argmax(self.test_accs3)])
		std_dev = 'standard deviation: {:.3}\n'.format(np.mean(np.std(self.test_probs, 0)))
		ex_time = "execution time is {}".format(self.sec2clock(self.train_time))
		self.statistics = (iters + latest_acc + top_acc + baseline + pass_test + test_acc2 + test_acc3 + std_dev + ex_time)
		return self.statistics

	def update_dataset_baselines(self):
		'''
		Updates the dataset baselines
		'''
		self.data_set_baseline = self.baseline_pf(-1, 0)[0]
##		self.top_3_baseline = np.sum(np.sort(np.bincount(self.data_set.sents_np))[::-1][:3])/self.data_set.size
		# sum the probabilities of the top 3 classes and divide by all
		self.top_3_baseline = np.sum(np.sort(np.sum(self.data_set.sents_sc_np, 0))[::-1][:3])/self.data_set.size
		# sum the counts of the top 3 classes and divide by all
		self.top_3_baseline0 = np.count_nonzero(self.data_set.sents_mh_np[:, np.argsort(np.count_nonzero(self.data_set.sents_sc_np, 0))[::-1][:3]])/self.data_set.size

	def update_data_set_statistics(self):
		'''
		Update the statistics of the data set
		'''
		self.update_dataset_baselines()
		classes = 'Distribution:\n\tSize: {}\n\tmax input length: {}\n\tTrain patterns: {}\tTest patters: {}\n\tClass counts: {}\n\tTweets with multiple feelings: {} ({:.3}%)'.format(
			self.data_set.size, self.data_set.max_length, self.max_train, self.max_test, self.data_set.class_counts, self.data_set.multiclass_count, self.data_set.multiclass_ratio*100)
		baselines = 'Baselines:\n\tData-set baseline: {:.3}\n\tTop-3-outputs-include-target baseline: {:.3}, baseline0: {:.3}'.format(
			self.data_set_baseline, self.top_3_baseline, self.top_3_baseline0)
		unmatched_words = '\n'.join(['\t\t{}: {}'.format(item[0], item[1]) for item in self.data_set.unmatched_words_counts])
		matching = 'Word matching:\n\tMatch ratio: {:.5}\n\tTotal Unmatched words: {}\n\tUnmatched words:\n{}'.format(
			self.data_set.match_ratio, self.data_set.unmatched_words_count, unmatched_words)
		self.data_set_statistics = '\n\n'.join([classes, baselines, matching])

	def set_plot(self):
		'''
		Prepares the plot of the train and test accuracies
		'''
		plt.gca().cla() 
		plt.plot(self.saved_iters, self.train_accs, label="train(bl=" + '{:.3}'.format(self.train_baseline) + ")")
		plt.plot(self.saved_iters, self.test_accs, label="test (bl=" + '{:.3}'.format(self.test_baseline) + ")")
		plt.title('Accuracy at iterations ({}:{})'.format(self.saved_iters[0], self.saved_iters[-1]))
		plt.legend(loc='best') #upper left

	def tweets_with_results(self, group ='wrong', threshold = 0.5):
		'''
		Gets each tweet with its results from the classifier and targets as a string
		'''
		sent_map=['Happy','Love','Hopeful','Neutral','Angry','Hopeless','Hate','Sad']
		text = ''
		for i in range(self.max_test):
			item_acc = np.sum(self.tests_acc[i])/np.sum(self.data_set.sents_sc_np[self.max_train+i])
			condition = True
			if group == 'wrong': condition = item_acc<threshold
			if group == 'correct': condition = item_acc>=threshold
			if condition:
				tweet = self.data_set.tweets[self.max_train+i]
				ress=', '.join(['{}: {:.3}'.format(sent_map[l], self.test_probs[i, l]) for l in np.argsort(self.test_probs[i])[::-1][:3] if self.test_probs[i, l]>0.01])
				targets=', '.join([sent_map[j] for j in self.data_set.sentiments_lists[self.max_train+i]])
				text += tweet + '\n> ' + ress + " >> " + targets + '\n'
		return text

	def log(self):
		'''
		Saves a log about the training accuracies and results and graph description
		'''
		if not os.path.exists(self.log_folder): os.makedirs(self.log_folder)
		
		plot_file_name = os.path.join(self.log_folder, "plot")
		self.set_plot()
		plt.savefig(plot_file_name + '.svg');
		plt.savefig(plot_file_name + '.png');
		plt.close()
		
		stats = self.statistics + ('\n\nsession save path:\n{}'.format(self.sess_save_path) if self.sess_save_path is not None else '')
		with open(os.path.join(self.log_folder, "statistics.txt"), 'w') as log_file: log_file.write(stats)
		
		with open(os.path.join(self.log_folder, "results-rejected.txt"), 'w', encoding = 'utf-8') as log_file:
			log_file.write(self.tweets_with_results())
		with open(os.path.join(self.log_folder, "results-approved.txt"), 'w', encoding = 'utf-8') as log_file:
			log_file.write(self.tweets_with_results(group = 'correct'))
		if self.note!='':
			with open(os.path.join(self.log_folder, "note.txt"), 'w') as log_file: log_file.write(self.note)
		if len(self.saved_iters)==1:
			with open(os.path.join(self.log_folder, "graph_description.txt"), 'w') as log_file:
				log_file.write(json.dumps(self.graph_description, indent=4, sort_keys=True))
			if self.script_path is not None:
				shutil.copy2(self.script_path, self.log_folder)
			with open(os.path.join(self.log_folder, "data_set_statistics.txt"), 'w', encoding='utf-8') as log_file:
				log_file.write(self.data_set_statistics)

	def set_note(self):
		'''
		Set a note to be saved with the log
		'''
		self.note = input("Note:\n")
		
	def sec2clock(self, s):
		'''
		Convert seconds to digital clock like format (88:88:88 or 8 seconds)
		args:
		s: seconds
		'''
		m, s = divmod(s, 60)
		h, m = divmod(m, 60)
		hours = '{:2}:'.format(int(h)) if h else ''
		minutes = '{:2}:'.format(int(m)) if m or h else ''
		seconds = '{:2.0f}'.format(s)
		if not hours and not minutes: seconds+= ' seconds'
		return (hours + minutes + seconds).strip()
